#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

class PromptMLP(nn.Module):
    """
    P-tuning的PromptMLP模块 - 对应论文2.2节

    使用10个[unused] token作为虚拟token，通过两层MLP映射为连续向量
    每个任务拥有独立的PromptMLP
    """

    def __init__(self, hidden_size=768, prompt_length=10, dropout_rate=0.3):
        super(PromptMLP, self).__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size)
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.prompt_embeddings)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self):
        x = self.prompt_embeddings
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        return x


class CitationPromptModel(nn.Module):

    def __init__(
        self,
        model_name='allenai/scibert_scivocab_uncased',
        model_dir=None,
        prompt_length=10,
        hidden_size=768,
        dropout_rate=0.3,
        alpha=1e-5
    ):
        super().__init__()

        load_path = model_dir if model_dir else model_name

        # 改成 MLM
        self.bert = AutoModelForMaskedLM.from_pretrained(
            load_path,
            local_files_only=True
        )
        self.resize_position_embeddings(
            prompt_length
            )
        # 冻结 SciBERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.alpha = alpha

        # 三个任务独立 Prompt
        self.prompt_mlp_intent = PromptMLP(
            hidden_size,
            prompt_length,
            dropout_rate
        )

        self.prompt_mlp_section = PromptMLP(
            hidden_size,
            prompt_length,
            dropout_rate
        )

        self.prompt_mlp_worthiness = PromptMLP(
            hidden_size,
            prompt_length,
            dropout_rate
        )
    def resize_position_embeddings(
        self,
        prompt_length
    ):

        old_embeddings = self.bert.bert.embeddings.position_embeddings

        old_num, dim = old_embeddings.weight.shape

        new_num = old_num + prompt_length

        new_embeddings = nn.Embedding(
            new_num,
            dim
        )

    # 复制旧参数
        new_embeddings.weight.data[:old_num] = (
            old_embeddings.weight.data
        )

    # 新位置初始化
        new_embeddings.weight.data[old_num:] = (
            old_embeddings.weight.data[-1]
            .unsqueeze(0)
            .repeat(prompt_length, 1)
        )

        self.bert.bert.embeddings.position_embeddings = (
            new_embeddings
        )

        self.bert.config.max_position_embeddings = new_num
    def get_trainable_parameters(self):

        params = []

        params.extend(self.prompt_mlp_intent.parameters())
        params.extend(self.prompt_mlp_section.parameters())
        params.extend(self.prompt_mlp_worthiness.parameters())

        return params

    def build_inputs_embeds(
        self,
        input_ids,
        token_type_ids,
        prompt_embeddings
        ):

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        device = input_ids.device

    # ===== 原token embedding =====

        raw_embeddings = self.bert.bert.embeddings.word_embeddings(
            input_ids
    )

    # ===== prompt =====

        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
    )

    # ===== 拼接 =====

        inputs_embeds = torch.cat(
            [
                prompt_embeddings,
                raw_embeddings
            ],
            dim=1
        )

    # ===== position ids =====

        total_len = self.prompt_length + seq_len

        position_ids = torch.arange(
            total_len,
            device=device
        ).unsqueeze(0)

        position_embeddings = self.bert.bert.embeddings.position_embeddings(
            position_ids
        )

        inputs_embeds = inputs_embeds + position_embeddings

    # ===== token type =====

        token_type_ids = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    self.prompt_length,
                    device=device,
                    dtype=torch.long
                ),
                token_type_ids
            ],
            dim=1
        )

        token_type_embeddings = self.bert.bert.embeddings.token_type_embeddings(
            token_type_ids
    )

        inputs_embeds = inputs_embeds + token_type_embeddings

    # ===== LN + dropout =====

        inputs_embeds = self.bert.bert.embeddings.LayerNorm(
            inputs_embeds
    )

        inputs_embeds = self.bert.bert.embeddings.dropout(
            inputs_embeds
    )

        return inputs_embeds

    def build_attention_mask(self, attention_mask):

        batch_size = attention_mask.size(0)

        prompt_mask = torch.ones(
            batch_size,
            self.prompt_length,
            device=attention_mask.device
        )

        full_mask = torch.cat(
            [
                prompt_mask,
                attention_mask
            ],
            dim=1
        )

        return full_mask

    def get_mask_position(self, input_ids, mask_token_id):

        mask_positions = (
            input_ids == mask_token_id
        ).nonzero(as_tuple=True)

        return mask_positions

    def forward_single_task(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        prompt_encoder
    ):

        batch_size = input_ids.size(0)

    # ==================================================
    # soft prompt
    # ==================================================

        prompt_embeddings = prompt_encoder()

    # ==================================================
    # inputs embeds
    # ==================================================

        inputs_embeds = self.build_inputs_embeds(
            input_ids,
            token_type_ids,
            prompt_embeddings
        )

    # ==================================================
    # attention mask
    # ==================================================

        full_attention_mask = self.build_attention_mask(
            attention_mask
        )

    # ==================================================
    # token type ids
    # 必须扩展
    # ==================================================

        prompt_token_type_ids = torch.zeros(
            batch_size,
            self.prompt_length,
            dtype=torch.long,
            device=input_ids.device
        )

        full_token_type_ids = torch.cat(
            [
                prompt_token_type_ids,
                token_type_ids
            ],
            dim=1
        )

    # ==================================================
    # MLM forward
    # ==================================================

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            token_type_ids=full_token_type_ids
        )

        logits = outputs.logits

    # ==================================================
    # MASK位置
    # ==================================================

        mask_token_id = 103

        mask_positions = (
            input_ids == mask_token_id
        ).nonzero(as_tuple=True)

        batch_indices = mask_positions[0]

        seq_indices = (
            mask_positions[1]
            + self.prompt_length
        )

        mask_logits = logits[
            batch_indices,
            seq_indices,
            :
        ]

        return mask_logits

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids
    ):

        logits_intent = self.forward_single_task(
            input_ids,
            attention_mask,
            token_type_ids,
            self.prompt_mlp_intent
        )

        logits_section = self.forward_single_task(
            input_ids,
            attention_mask,
            token_type_ids,
            self.prompt_mlp_section
        )

        logits_worthiness = self.forward_single_task(
            input_ids,
            attention_mask,
            token_type_ids,
            self.prompt_mlp_worthiness
        )

        return {
            "intent": logits_intent,
            "section": logits_section,
            "worthiness": logits_worthiness
        }
    def compute_loss(
        self,
        logits,
        labels
    ):

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=-1
        )
    def compute_soft_sharing_loss(self):

        loss = 0.0

        prompt_modules = [
            self.prompt_mlp_intent,
            self.prompt_mlp_section,
            self.prompt_mlp_worthiness
        ]

        for i in range(len(prompt_modules)):
            for j in range(i + 1, len(prompt_modules)):

                p1 = prompt_modules[i].prompt_embeddings
                p2 = prompt_modules[j].prompt_embeddings

                loss += torch.norm(p1 - p2, p=2)

        return loss