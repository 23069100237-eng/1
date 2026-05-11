#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
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

class TaskHead(nn.Module):
    """
    任务头模块 - 每个任务独立的线性层映射到词表大小
    """

    def __init__(self, hidden_size=768, vocab_size=30522, task_type='classification'):
        super(TaskHead, self).__init__()
        self.task_type = task_type

        if task_type == 'classification':
            self.classifier = nn.Linear(hidden_size, vocab_size)
        elif task_type == 'worthiness':
            self.classifier = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        logits = self.classifier(x)
        return logits

class CitationPromptModel(nn.Module):
    """
    完整的多任务提示学习模型 - 对应论文图2框架
    """

    def __init__(
        self,
        model_name='allenai/scibert_scivocab_uncased',
        model_dir=None,
        prompt_length=10,
        hidden_size=768,
        dropout_rate=0.3,
        alpha=1e-5
    ):
        super(CitationPromptModel, self).__init__()

        load_path = model_dir if model_dir else model_name

        self.bert = AutoModel.from_pretrained(
            load_path,
            local_files_only=True
        )
        
        # 扩展位置嵌入以支持prompt tokens
        self._extend_position_embeddings(prompt_length)
        
        self._freeze_bert()

        self.vocab_size = self.bert.config.vocab_size

        self.prompt_mlp_intent = PromptMLP(
            hidden_size=hidden_size,
            prompt_length=prompt_length,
            dropout_rate=dropout_rate
        )
        self.prompt_mlp_section = PromptMLP(
            hidden_size=hidden_size,
            prompt_length=prompt_length,
            dropout_rate=dropout_rate
        )
        self.prompt_mlp_worthiness = PromptMLP(
            hidden_size=hidden_size,
            prompt_length=prompt_length,
            dropout_rate=dropout_rate
        )

        self.head_intent = TaskHead(
            hidden_size=hidden_size,
            vocab_size=self.vocab_size,
            task_type='classification'
        )
        self.head_section = TaskHead(
            hidden_size=hidden_size,
            vocab_size=self.vocab_size,
            task_type='classification'
        )
        self.head_worthiness = TaskHead(
            hidden_size=hidden_size,
            vocab_size=self.vocab_size,
            task_type='worthiness'
        )

        self.alpha = alpha
        # 兼容不同版本的配置，mask_token_id可能不存在
        self.mask_token_id = getattr(self.bert.config, 'mask_token_id', 103)

        # 扩展位置嵌入以支持prompt tokens
        self._extend_position_embeddings(prompt_length)

    def _extend_position_embeddings(self, prompt_length):
        """扩展位置嵌入以支持prompt tokens"""
        max_position_embeddings = self.bert.config.max_position_embeddings
        new_max_len = max_position_embeddings + prompt_length

        # 保存原始的position_embeddings
        original_position_embeddings = self.bert.embeddings.position_embeddings

        # 创建新的位置嵌入（扩展后的）
        new_embeddings = torch.nn.Embedding(new_max_len, self.bert.config.hidden_size)
        new_embeddings.weight.data[:max_position_embeddings] = original_position_embeddings.weight.data
        if new_max_len > max_position_embeddings:
            new_embeddings.weight.data[max_position_embeddings:] = original_position_embeddings.weight.data[-1].unsqueeze(0).repeat(prompt_length, 1)

        # 替换BERT的位置嵌入
        self.bert.embeddings.position_embeddings = new_embeddings
        self.bert.config.max_position_embeddings = new_max_len

    def _freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        params = []
        params.extend(self.prompt_mlp_intent.parameters())
        params.extend(self.prompt_mlp_section.parameters())
        params.extend(self.prompt_mlp_worthiness.parameters())
        params.extend(self.head_intent.parameters())
        params.extend(self.head_section.parameters())
        params.extend(self.head_worthiness.parameters())
        return params

    def _build_embeddings(self, input_ids, attention_mask, token_type_ids, task_prompt):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)

        embeddings = self.bert.embeddings.word_embeddings(input_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)

        position_ids = torch.arange(seq_length, device=input_ids.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)

        base_embeddings = embeddings + token_type_embeddings + position_embeddings
        base_embeddings = self.bert.embeddings.LayerNorm(base_embeddings)
        base_embeddings = self.bert.embeddings.dropout(base_embeddings)

        mask_embedding = self.bert.embeddings.word_embeddings(
            torch.tensor([self.mask_token_id], device=input_ids.device)
        )

        prompt_embeddings = task_prompt.unsqueeze(0).repeat(batch_size, 1, 1)

        cls_embedding = base_embeddings[:, 0:1, :]
        text_embeddings = base_embeddings[:, 1:-1, :]
        sep_embedding = base_embeddings[:, -1:, :]

        full_embeddings = torch.cat([
            prompt_embeddings,
            cls_embedding,
            text_embeddings,
            sep_embedding,
            mask_embedding.unsqueeze(0).repeat(batch_size, 1, 1),
            sep_embedding
        ], dim=1)

        prompt_mask = torch.ones(batch_size, task_prompt.size(0), device=input_ids.device)
        mask_attention = torch.ones(batch_size, 2, device=input_ids.device)
        attention_mask_full = torch.cat([
            prompt_mask,
            attention_mask,
            mask_attention
        ], dim=1)

        prompt_type_ids = torch.zeros(batch_size, task_prompt.size(0), device=input_ids.device, dtype=torch.long)
        mask_type_ids = torch.zeros(batch_size, 2, device=input_ids.device, dtype=torch.long)
        token_type_ids_full = torch.cat([
            prompt_type_ids,
            token_type_ids.long(),
            mask_type_ids
        ], dim=1)

        return (
            full_embeddings.detach().clone(),
            attention_mask_full,
            token_type_ids_full
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        prompt_intent = self.prompt_mlp_intent()
        prompt_section = self.prompt_mlp_section()
        prompt_worthiness = self.prompt_mlp_worthiness()

        emb_intent, mask_intent, type_intent = self._build_embeddings(
            input_ids, attention_mask, token_type_ids, prompt_intent
        )
        emb_section, mask_section, type_section = self._build_embeddings(
            input_ids, attention_mask, token_type_ids, prompt_section
        )
        emb_worthiness, mask_worthiness, type_worthiness = self._build_embeddings(
            input_ids, attention_mask, token_type_ids, prompt_worthiness
        )

        mask_position = prompt_intent.size(0) + input_ids.size(1) - 1

        # 正确处理 attention mask
        extended_attention_mask = mask_intent[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        outputs_intent = self.bert.encoder(
            hidden_states=emb_intent,
            attention_mask=extended_attention_mask
        )

        mask_hidden_intent = outputs_intent.last_hidden_state[:, mask_position, :]
        logits_intent = self.head_intent(mask_hidden_intent)

        outputs_section = self.bert.encoder(
            hidden_states=emb_section,
            attention_mask=extended_attention_mask
        )

        mask_hidden_section = outputs_section.last_hidden_state[:, mask_position, :]
        logits_section = self.head_section(mask_hidden_section)

        outputs_worthiness = self.bert.encoder(
            hidden_states=emb_worthiness,
            attention_mask=extended_attention_mask
        )

        mask_hidden_worthiness = outputs_worthiness.last_hidden_state[:, mask_position, :]
        logits_worthiness = self.head_worthiness(mask_hidden_worthiness)

        return {
            'intent': logits_intent,
            'section': logits_section,
            'worthiness': logits_worthiness
        }

    def compute_loss(
        self,
        logits_intent, logits_section, logits_worthiness,
        labels_intent, labels_section, labels_worthiness,
        lambda_int=1.0, lambda_sec=0.16, lambda_wor=0.32
    ):
        loss_intent = F.cross_entropy(
            logits_intent, labels_intent,
            ignore_index=-1,
            reduction='mean'
        )

        loss_section = F.cross_entropy(
            logits_section, labels_section,
            ignore_index=-1,
            reduction='mean'
        )

        loss_worthiness = F.binary_cross_entropy_with_logits(
            logits_worthiness.squeeze(),
            labels_worthiness,
            reduction='mean'
        )

        l2_reg = 0.0
        for param in self.prompt_mlp_intent.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        for param in self.prompt_mlp_section.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        for param in self.prompt_mlp_worthiness.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        for param in self.head_intent.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        for param in self.head_section.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        for param in self.head_worthiness.parameters():
            l2_reg += torch.norm(param, p=2) ** 2

        total_loss = (
            lambda_int * loss_intent +
            lambda_sec * loss_section +
            lambda_wor * loss_worthiness +
            self.alpha * l2_reg
        )

        return {
            'total': total_loss,
            'intent': loss_intent,
            'section': loss_section,
            'worthiness': loss_worthiness,
            'l2_reg': self.alpha * l2_reg
        }

    def predict(self, logits, label_expansion, tokenizer):
        probs = F.softmax(logits, dim=1)

        num_labels = len(label_expansion)
        batch_size = logits.size(0)
        label_probs = torch.zeros(batch_size, num_labels, device=logits.device)

        for label_idx, (label, words) in enumerate(label_expansion.items()):
            word_ids = []
            for word in words:
                token = tokenizer.encode(word, add_special_tokens=False)
                if len(token) == 1:
                    word_ids.append(token[0])

            if word_ids:
                label_probs[:, label_idx] = probs[:, word_ids].mean(dim=1)
            else:
                label_probs[:, label_idx] = 0.0

        predictions = torch.argmax(label_probs, dim=1)

        return predictions, label_probs