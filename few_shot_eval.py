"""
few_shot_train_eval.py

真正的 Few-Shot 版本
支持：

1-shot
5-shot
10-shot
20-shot

方式：
每类抽 K 个样本重新训练 Prompt

运行：

python few_shot_train_eval.py \
    --shots 1 5 10 20
"""

import argparse
import json
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

from utils import (
    setup_logging
)


# =========================================================
# 固定随机种子
# =========================================================

def set_seed(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =========================================================
# Few-shot sampling
# =========================================================

def build_few_shot_dataset(
    dataset,
    shot_num
):

    label_buckets = {
        'Background': [],
        'Method': [],
        'Result': []
    }

    for item in dataset.data:

        raw_label = item.get(
            'label',
            ''
        ).lower()

        if raw_label == 'background':

            label = 'Background'

        elif raw_label == 'method':

            label = 'Method'

        elif raw_label == 'result':

            label = 'Result'

        else:

            continue

        label_buckets[label].append(item)

    sampled_data = []

    for label in label_buckets:

        sampled = random.sample(
            label_buckets[label],
            min(
                shot_num,
                len(label_buckets[label])
            )
        )

        sampled_data.extend(sampled)

    return sampled_data


# =========================================================
# train
# =========================================================

def train_few_shot(
    model,
    tokenizer,
    train_data,
    verbalizer,
    device
):

    model.train()

    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=LEARNING_RATE
    )

    label_map = {
        'background': 0,
        'method': 1,
        'result': 2
    }

    for epoch in range(5):

        total_loss = 0.0

        progress_bar = tqdm(
            train_data,
            desc=f"train epoch {epoch+1}"
        )

        for item in progress_bar:

            text = item.get(
                'string',
                ''
            )

            raw_label = item.get(
                'label',
                ''
            ).lower()

            if raw_label not in label_map:

                continue

            label_id = label_map[raw_label]

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

            input_ids = encoding[
                'input_ids'
            ].to(device)

            attention_mask = encoding[
                'attention_mask'
            ].to(device)

            token_type_ids = encoding[
                'token_type_ids'
            ].to(device)

            mask_logits = model.forward_single_task(
                input_ids,
                attention_mask,
                token_type_ids,
                model.prompt_mlp_intent
            )

            class_logits = verbalizer.project(
                mask_logits
            )

            labels = torch.tensor(
                [label_id],
                dtype=torch.long,
                device=device
            )

            loss = F.cross_entropy(
                class_logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_data)

        print(
            f"epoch {epoch+1} "
            f"loss={avg_loss:.4f}"
        )


# =========================================================
# evaluate
# =========================================================

def evaluate(
    model,
    dataloader,
    verbalizer,
    device,
    intent_labels
):

    model.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="evaluating"
    )

    with torch.no_grad():

        for batch in progress_bar:

            input_ids = (
                batch['input_ids']
                .to(device)
            )

            attention_mask = (
                batch['attention_mask']
                .to(device)
            )

            token_type_ids = (
                batch['token_type_ids']
                .to(device)
            )

            labels = (
                batch['intent_label']
                .to(device)
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

            predictions = torch.argmax(
                logits,
                dim=1
            )

            for pred, label in zip(
                predictions,
                labels
            ):

                if label.item() == -1:
                    continue

                pred_idx = pred.item()

                # 防止越界
                pred_idx = min(
                    pred_idx,
                    len(intent_labels) - 1
                )

                pred_label = intent_labels[
                    pred_idx
                ]

                true_label = intent_labels[
                    label.item()
                ]

                if pred_label == true_label:
                    correct += 1

                total += 1

            progress_bar.set_postfix({
                'acc': f"{correct / max(total,1):.4f}"
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--shots',
        type=int,
        nargs='+',
        default=[1, 5, 10, 20]
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    args = parser.parse_args()

    set_seed(42)

    logger = setup_logging(
        'few_shot.log'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    train_dataset = CitationDataset(
        DATA_FILES['scicite']['train'],
        tokenizer,
        max_len=MAX_LEN,
        dataset_type='scicite'
    )

    test_dataset = CitationDataset(
        DATA_FILES['scicite']['test'],
        tokenizer,
        max_len=MAX_LEN,
        dataset_type='scicite'
    )

    label_expansions = {
        'intent':
        LabelExpansionDict.get_intent_expansion()
    }

    verbalizer = Verbalizer(
        tokenizer,
        label_expansions['intent']
    )

    results = {}

    print("\n" + "=" * 60)

    print("Few-Shot Results")

    print("=" * 60)

    for shot in args.shots:

        print(f"\n{shot}-shot")

        few_shot_data = build_few_shot_dataset(
            train_dataset,
            shot
        )

        model = CitationPromptModel(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,
            prompt_length=PROMPT_LENGTH,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            alpha=L2_ALPHA
        )

        model = model.to(args.device)

        train_few_shot(
            model,
            tokenizer,
            few_shot_data,
            verbalizer,
            args.device
        )

        accuracy = evaluate(
            model,
            test_dataset,
            tokenizer,
            verbalizer,
            args.device
        )

        results[shot] = accuracy

        print(
            f"{shot}-shot accuracy: "
            f"{accuracy:.4f}"
        )

    print("\n" + "=" * 60)

    print("Summary")

    print("=" * 60)

    for shot in results:

        print(
            f"{shot}-shot: "
            f"{results[shot]:.4f}"
        )

    with open(
        'few_shot_results.json',
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )


if __name__ == '__main__':

    main()