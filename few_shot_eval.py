"""
少样本验证脚本 - 支持0-shot、1-shot、5-shot、10-shot、20-shot
"""

import argparse
import json
import random
import torch

from transformers import AutoTokenizer
from tqdm import tqdm

from data import CitationDataset, LabelExpansionDict
from model import CitationPromptModel
from verbalizer import Verbalizer
from utils import setup_logging

from config import *

random.seed(42)

# =========================================================
# build few-shot prompt
# =========================================================

def create_few_shot_prompt(
    tokenizer,
    examples,
    test_text,
    max_len=512
):

    prompt = ""

    for example in examples:

        prompt += (
            f"Citation: {example['text']}\n"
            f"Intent: {example['label']}\n\n"
        )

    prompt += (
        f"Citation: {test_text}\n"
        f"Intent: {tokenizer.mask_token}"
    )

    encoding = tokenizer(

        prompt,

        truncation=True,

        max_length=max_len,

        padding='max_length',

        return_attention_mask=True,

        return_token_type_ids=True,

        return_tensors='pt'
    )

    return encoding


# =========================================================
# few-shot eval
# =========================================================

def evaluate_few_shot(

    model,

    test_dataset,

    train_dataset,

    tokenizer,

    shots,

    device,

    label_expansion
):

    model.eval()

    verbalizer = Verbalizer(
        tokenizer,
        label_expansion['intent']
    )

    intent_labels = list(
        label_expansion['intent'].keys()
    )

    # =====================================================
    # build support set
    # =====================================================

    examples = []

    if shots > 0:

        train_by_label = {
            label: []
            for label in intent_labels
        }

        for item in train_dataset.data:

            if train_dataset.dataset_type == 'scicite':

                raw_label = item.get(
                    'label',
                    ''
                ).lower()

                label_map = {

                    'background': 'Background',

                    'method': 'Method',

                    'result': 'Result'
                }

                label = label_map.get(
                    raw_label,
                    None
                )

                text = item.get(
                    'string',
                    ''
                )

            else:

                label = item.get(
                    'intent',
                    None
                )

                text = item.get(
                    'text',
                    ''
                )

            if (
                label in train_by_label
                and
                text
            ):

                train_by_label[label].append({

                    'text': text,

                    'label': label
                })

        for label in intent_labels:

            available = train_by_label[label]

            if len(available) == 0:
                continue

            selected = random.sample(

                available,

                min(shots, len(available))
            )

            examples.extend(selected)

        random.shuffle(examples)

    # =====================================================
    # evaluation
    # =====================================================

    correct = 0

    total = 0

    progress_bar = tqdm(
        test_dataset.data,
        desc=f"{shots}-shot"
    )

    with torch.no_grad():

        for item in progress_bar:

            # =================================================
            # text + label
            # =================================================

            if test_dataset.dataset_type == 'scicite':

                test_text = item.get(
                    'string',
                    ''
                )

                raw_label = item.get(
                    'label',
                    ''
                ).lower()

                label_map = {

                    'background': 'Background',

                    'method': 'Method',

                    'result': 'Result'
                }

                true_label = label_map.get(
                    raw_label,
                    None
                )

            else:

                test_text = item.get(
                    'text',
                    ''
                )

                true_label = item.get(
                    'intent',
                    None
                )

            if (
                not test_text
                or
                true_label not in intent_labels
            ):

                continue

            # =================================================
            # build prompt
            # =================================================

            encoding = create_few_shot_prompt(

                tokenizer,

                examples,

                test_text,

                MAX_LEN
            )

            input_ids = (
                encoding['input_ids']
                .to(device)
            )

            attention_mask = (
                encoding['attention_mask']
                .to(device)
            )

            token_type_ids = (
                encoding['token_type_ids']
                .to(device)
            )

            # =================================================
            # forward
            # =================================================

            mask_logits = model.forward_single_task(

                input_ids,

                attention_mask,

                token_type_ids,

                model.prompt_mlp_intent
            )

            logits = verbalizer.project(
                mask_logits
            )

            pred_idx = torch.argmax(
                logits,
                dim=1
            ).item()

            pred_label = intent_labels[pred_idx]

            # =================================================
            # metrics
            # =================================================

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


# =========================================================
# main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        '--model_path',

        type=str,

        default='./output/best_model.pt'
    )

    parser.add_argument(

        '--shots',

        type=int,

        nargs='+',

        default=[0, 1, 5, 10, 20]
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

    # =====================================================
    # tokenizer
    # =====================================================

    tokenizer = AutoTokenizer.from_pretrained(

        MODEL_DIR,

        local_files_only=True
    )

    # =====================================================
    # labels
    # =====================================================

    label_expansions = {

        'intent':
            LabelExpansionDict.get_intent_expansion(),

        'section':
            LabelExpansionDict.get_section_expansion(),

        'worthiness':
            LabelExpansionDict.get_worthiness_expansion()
    }

    # =====================================================
    # dataset
    # =====================================================

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

    # =====================================================
    # model
    # =====================================================

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

    # =====================================================
    # eval
    # =====================================================

    results = {}

    print("\n" + "=" * 60)

    print(f"{args.dataset} Few-Shot Results")

    print("=" * 60)

    for shots in args.shots:

        logger.info(
            f"开始 {shots}-shot"
        )

        accuracy = evaluate_few_shot(

            model,

            test_dataset,

            train_dataset,

            tokenizer,

            shots,

            args.device,

            label_expansions
        )

        results[shots] = accuracy

        print(f"\n{shots}-shot")

        print(f"Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)

    print("Summary")

    print("=" * 60)

    for shots in sorted(results.keys()):

        print(
            f"{shots}-shot: "
            f"{results[shots]:.4f}"
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

    logger.info("few-shot eval finished")


if __name__ == '__main__':

    main()
