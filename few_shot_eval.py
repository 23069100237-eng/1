# few_shot_eval.py

import random
import copy
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer

from data import CitationDataset, LabelExpansionDict
from model import CitationPromptModel
from verbalizer import Verbalizer
from config import *

# =========================================================
# config
# =========================================================

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

SHOTS = [1, 5, 10, 20]

EPOCHS = 5

LR = 1e-3

# =========================================================
# dataset
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

train_dataset = CitationDataset(
    DATA_FILES['scicite']['train'],
    tokenizer,
    MAX_LEN,
    'scicite'
)

test_dataset = CitationDataset(
    DATA_FILES['scicite']['test'],
    tokenizer,
    MAX_LEN,
    'scicite'
)

# =========================================================
# verbalizer
# =========================================================

label_expansions = {
    'intent': LabelExpansionDict.get_intent_expansion()
}

verbalizer = Verbalizer(
    tokenizer,
    label_expansions['intent']
)

intent_labels = list(
    label_expansions['intent'].keys()
)

# =========================================================
# build few-shot dataset
# =========================================================

def build_few_shot_dataset(
    dataset,
    shot
):

    label_map = {
        'background': [],
        'method': [],
        'result': []
    }

    for item in dataset.data:

        label = str(
            item.get('label', '')
        ).lower()

        if label in label_map:
            label_map[label].append(item)

    selected = []

    for label in label_map:

        samples = random.sample(
            label_map[label],
            min(
                shot,
                len(label_map[label])
            )
        )

        selected.extend(samples)

    random.shuffle(selected)

    return selected

# =========================================================
# training
# =========================================================

def train_few_shot(
    model,
    few_shot_data
):

    model.train()

    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=LR
    )

    for epoch in range(EPOCHS):

        total_loss = 0.0

        progress_bar = tqdm(
            few_shot_data,
            desc=f"train epoch {epoch+1}"
        )

        for item in progress_bar:

            text = item['string']

            label = str(
                item['label']
            ).lower()

            label_map = {
                'background': 0,
                'method': 1,
                'result': 2
            }

            if label not in label_map:
                continue

            label_id = label_map[label]

            prompt_text = (
                text
                + " "
                + tokenizer.mask_token
                + "."
            )

            encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=MAX_LEN - PROMPT_LENGTH,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            input_ids = (
                encoding['input_ids']
                .to(DEVICE)
            )

            attention_mask = (
                encoding['attention_mask']
                .to(DEVICE)
            )

            token_type_ids = (
                encoding['token_type_ids']
                .to(DEVICE)
            )

            label_tensor = torch.tensor(
                [label_id],
                dtype=torch.long,
                device=DEVICE
            )

            optimizer.zero_grad()

            mask_logits = model.forward_single_task(
                input_ids,
                attention_mask,
                token_type_ids,
                model.prompt_mlp_intent
            )

            logits = verbalizer.project(
                mask_logits
            )

            # 只保留前三类
            logits = logits[:, :3]

            loss = F.cross_entropy(
                logits,
                label_tensor
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

        avg_loss = (
            total_loss / max(len(few_shot_data), 1)
        )

        print(
            f"epoch {epoch+1} "
            f"loss={avg_loss:.4f}"
        )

# =========================================================
# evaluation
# =========================================================

def evaluate(
    model,
    dataset
):

    model.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(
        dataset.data,
        desc="evaluating"
    )

    label_map = {
        'background': 'Background',
        'method': 'Method',
        'result': 'Result'
    }

    with torch.no_grad():

        for item in progress_bar:

            text = item['string']

            raw_label = str(
                item['label']
            ).lower()

            if raw_label not in label_map:
                continue

            true_label = label_map[
                raw_label
            ]

            prompt_text = (
                text
                + " "
                + tokenizer.mask_token
                + "."
            )

            encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=MAX_LEN - PROMPT_LENGTH,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            input_ids = (
                encoding['input_ids']
                .to(DEVICE)
            )

            attention_mask = (
                encoding['attention_mask']
                .to(DEVICE)
            )

            token_type_ids = (
                encoding['token_type_ids']
                .to(DEVICE)
            )

            mask_logits = model.forward_single_task(
                input_ids,
                attention_mask,
                token_type_ids,
                model.prompt_mlp_intent
            )

            logits = verbalizer.project(
                mask_logits
            )

            logits = logits[:, :3]

            pred_idx = torch.argmax(
                logits,
                dim=1
            ).item()

            pred_idx = min(
                pred_idx,
                2
            )

            pred_label = [
                'Background',
                'Method',
                'Result'
            ][pred_idx]

            if pred_label == true_label:
                correct += 1

            total += 1

            progress_bar.set_postfix({
                'acc': f"{correct/max(total,1):.4f}"
            })

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return accuracy

# =========================================================
# main
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("Few-Shot Results")
    print("=" * 60)

    for shot in SHOTS:

        print(f"\n{shot}-shot")

        # =========================
        # init model
        # =========================

        model = CitationPromptModel(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            prompt_length=PROMPT_LENGTH,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            alpha=L2_ALPHA
        )

        checkpoint = torch.load(
            './output/best_model.pt',
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint['model_state_dict']
        )

        model = model.to(DEVICE)

        # =========================
        # few-shot data
        # =========================

        few_shot_data = build_few_shot_dataset(
            train_dataset,
            shot
        )

        # =========================
        # train
        # =========================

        train_few_shot(
            model,
            few_shot_data
        )

        # =========================
        # evaluate
        # =========================

        accuracy = evaluate(
            model,
            test_dataset
        )

        print(f"\nAccuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)

# =========================================================
# run
# =========================================================

if __name__ == '__main__':

    main()