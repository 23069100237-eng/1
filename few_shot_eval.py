"""
few_shot_eval.py
修正版
"""

import argparse
import json
import torch

from transformers import AutoTokenizer
from tqdm import tqdm

from data import CitationDataset, LabelExpansionDict
from model import CitationPromptModel
from verbalizer import Verbalizer

from config import *

from utils import setup_logging


def evaluate_few_shot(
    model,
    test_dataset,
    tokenizer,
    device,
    verbalizer,
    intent_labels
):

    model.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(
        test_dataset,
        desc="evaluating"
    )

    with torch.no_grad():

        for item in progress_bar:

            text = item.get('string', '')

            raw_label = item.get(
                'label',
                ''
            ).lower()

            # =========================
            # label normalize
            # =========================

            if raw_label == 'background':

                true_label = 'Background'

            elif raw_label == 'method':

                true_label = 'Method'

            elif raw_label == 'result':

                true_label = 'Result'

            else:

                continue

            # =========================
            # prompt text
            # =========================

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

            # =========================
            # MLM forward
            # =========================

            mask_logits = model.forward_single_task(
                input_ids,
                attention_mask,
                token_type_ids,
                model.prompt_mlp_intent
            )

            # =========================
            # verbalizer
            # =========================

            class_logits = verbalizer.project(
                mask_logits
            )

            pred_idx = torch.argmax(
                class_logits,
                dim=1
            ).item()

            pred_label = intent_labels[
                pred_idx
            ]

            # =========================
            # accuracy
            # =========================

            if pred_label == true_label:

                correct += 1

            total += 1

            progress_bar.set_postfix({
                'acc': f"{correct / total:.4f}"
            })

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return accuracy


def main():

    parser = argparse.ArgumentParser(
        description='Few Shot Evaluation'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./output/best_model.pt'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='scicite'
    )

    args = parser.parse_args()

    logger = setup_logging(
        'few_shot_eval.log'
    )

    logger.info(
        f"使用设备: {args.device}"
    )

    # =========================
    # tokenizer
    # =========================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    # =========================
    # dataset
    # =========================

    train_dataset = CitationDataset(
        DATA_FILES[args.dataset]['train'],
        tokenizer,
        max_len=MAX_LEN,
        dataset_type=args.dataset
    )

    test_dataset = CitationDataset(
        DATA_FILES[args.dataset]['test'],
        tokenizer,
        max_len=MAX_LEN,
        dataset_type=args.dataset
    )

    logger.info(
        f"train size: {len(train_dataset)}"
    )

    logger.info(
        f"test size: {len(test_dataset)}"
    )

    # =========================
    # model
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
        args.model_path,
        map_location=args.device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model = model.to(args.device)

    logger.info("模型加载完成")

    # =========================
    # verbalizer
    # =========================

    label_expansions = {
        'intent':
        LabelExpansionDict.get_intent_expansion()
    }

    intent_verbalizer = Verbalizer(
        tokenizer,
        label_expansions['intent']
    )

    intent_labels = list(
        label_expansions['intent'].keys()
    )

    # =========================
    # evaluate
    # =========================

    print("\n" + "=" * 60)

    print(
        f"{args.dataset} Evaluation Results"
    )

    print("=" * 60)

    accuracy = evaluate_few_shot(
        model=model,
        test_dataset=test_dataset.data,
        tokenizer=tokenizer,
        device=args.device,
        verbalizer=intent_verbalizer,
        intent_labels=intent_labels
    )

    print("\nAccuracy: "
          f"{accuracy:.4f}")

    print("=" * 60)

    # =========================
    # save
    # =========================

    results = {
        'accuracy': accuracy
    }

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

    logger.info("评估完成")


if __name__ == '__main__':

    main()