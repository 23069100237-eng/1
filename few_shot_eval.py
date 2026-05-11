"""
少样本验证脚本 - 支持0-shot、1-shot、5-shot、10-shot、20-shot

运行方式:
python few_shot_eval.py --model_path ./output/best_model.pt --shots 0 1 5 10 20
"""

import argparse
import torch
import random
from transformers import AutoTokenizer
from tqdm import tqdm

from data import CitationDataset, LabelExpansionDict
from model import CitationPromptModel
from utils import count_parameters, setup_logging
from config import *

def create_few_shot_prompt(tokenizer, examples, test_text, max_len=512):
    """
    创建少样本提示
    格式: [CLS] 示例1 [SEP] [MASK]=label1 [SEP] ... [CLS] 测试文本 [SEP] [MASK] [SEP]
    """
    prompt_parts = []

    for example in examples:
        text = example['text']
        label = example['label']

        example_encoding = tokenizer.encode(
            text,
            truncation=True,
            max_length=(max_len // (len(examples) + 1)) - 8,
            add_special_tokens=False
        )

        prompt_parts.extend([tokenizer.cls_token_id])
        prompt_parts.extend(example_encoding)
        prompt_parts.extend([tokenizer.sep_token_id])
        prompt_parts.extend([tokenizer.mask_token_id])

        label_tokens = tokenizer.encode(label.lower(), add_special_tokens=False)
        if label_tokens:
            prompt_parts.extend(label_tokens)
        prompt_parts.extend([tokenizer.sep_token_id])

    test_encoding = tokenizer.encode(
        test_text,
        truncation=True,
        max_length=max_len - len(prompt_parts) - 4,
        add_special_tokens=False
    )

    prompt_parts.extend([tokenizer.cls_token_id])
    prompt_parts.extend(test_encoding)
    prompt_parts.extend([tokenizer.sep_token_id])
    prompt_parts.extend([tokenizer.mask_token_id])
    prompt_parts.extend([tokenizer.sep_token_id])

    if len(prompt_parts) > max_len:
        prompt_parts = prompt_parts[:max_len]

    return torch.tensor(prompt_parts, dtype=torch.long)

def evaluate_few_shot(model, test_dataset, train_dataset, tokenizer, shots, device, label_expansion):
    """
    执行少样本评估
    """
    model.eval()

    intent_labels = list(label_expansion['intent'].keys())
    examples = []

    if shots > 0:
        train_by_label = {label: [] for label in intent_labels}

        for item in train_dataset:
            if 'intent' in item:
                label = item['intent']
            elif 'label' in item:
                label = item['label'].capitalize()

            if label in train_by_label:
                train_by_label[label].append({
                    'text': item.get('text', '') or item.get('string', ''),
                    'label': label
                })

        for label in intent_labels:
            available = train_by_label.get(label, [])
            selected = random.sample(available, min(shots, len(available)))
            examples.extend(selected)

        random.shuffle(examples)

    correct = 0
    total = 0

    progress_bar = tqdm(test_dataset, desc=f"{shots}-shot")

    with torch.no_grad():
        for item in progress_bar:
            if isinstance(item, dict):
                test_text = item.get('text', '') or item.get('string', '')
                raw_label = item.get('intent', '') or item.get('label', '')
            else:
                test_text = test_dataset.data[item].get('text', '') or test_dataset.data[item].get('string', '')
                raw_label = test_dataset.data[item].get('intent', '') or test_dataset.data[item].get('label', '')

            true_label = raw_label.capitalize() if isinstance(raw_label, str) else 'Unknown'

            if true_label not in intent_labels:
                continue

            encoding = tokenizer(
                test_text,
                truncation=True,
                max_length=MAX_LEN - 20,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            probs = torch.softmax(outputs['intent'], dim=1)
            label_probs = torch.zeros(len(intent_labels), device=device)

            for label_idx, (label, words) in enumerate(label_expansion['intent'].items()):
                word_ids = []
                for word in words:
                    token = tokenizer.encode(word, add_special_tokens=False)
                    if len(token) == 1:
                        word_ids.append(token[0])

                if word_ids:
                    label_probs[label_idx] = probs[0, word_ids].mean()

            pred_idx = torch.argmax(label_probs).item()
            pred_label = intent_labels[pred_idx]

            if pred_label == true_label:
                correct += 1
            total += 1

            progress_bar.set_postfix({
                'acc': f"{correct/total:.4f}",
                'correct': correct,
                'total': total
            })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='少样本验证脚本')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pt',
                        help='模型检查点路径')
    parser.add_argument('--shots', type=int, nargs='+', default=[0, 1, 5, 10, 20],
                        help='要测试的shot数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--dataset', type=str, default='scicite',
                        help='数据集类型')
    args = parser.parse_args()

    logger = setup_logging('few_shot_eval.log')

    logger.info(f"加载tokenizer: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    label_expansions = {
        'intent': LabelExpansionDict.get_intent_expansion(),
        'section': LabelExpansionDict.get_section_expansion(),
        'worthiness': LabelExpansionDict.get_worthiness_expansion()
    }

    logger.info(f"加载{args.dataset}数据集")
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

    logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    logger.info(f"加载模型: {args.model_path}")
    model = CitationPromptModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        prompt_length=PROMPT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        alpha=L2_ALPHA
    )

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    logger.info(f"模型已加载到 {args.device}")

    results = {}

    print(f"\n{'='*60}")
    print(f"少样本验证结果 ({args.dataset}数据集)")
    print(f"{'='*60}")

    for shots in args.shots:
        logger.info(f"开始{shots}-shot评估")
        accuracy = evaluate_few_shot(
            model, test_dataset.data, train_dataset.data,
            tokenizer, shots, args.device, label_expansions
        )
        results[shots] = accuracy

        print(f"\n{shots}-shot:")
        print(f"  准确率: {accuracy:.4f}")

    print(f"\n{'='*60}")
    print("汇总结果:")
    print(f"{'='*60}")
    for shots in sorted(results.keys()):
        print(f"{shots}-shot: {results[shots]:.4f}")

    import json
    with open('few_shot_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    logger.info("少样本评估完成")

if __name__ == '__main__':
    main()