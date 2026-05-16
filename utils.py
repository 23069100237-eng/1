"""
工具函数模块 - 包含评估指标、模型保存、日志记录等
"""

# utils.py

import json
import logging

import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from verbalizer import Verbalizer


def setup_logging(log_file='training.log'):
    """设置日志记录"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def compute_accuracy(logits, labels):
    """计算准确率"""

    predictions = torch.argmax(logits, dim=1)

    mask = labels != -1

    correct = (
        predictions[mask] == labels[mask]
    ).sum().item()

    total = mask.sum().item()

    return correct / total if total > 0 else 0.0


def evaluate_multitask(
    model,
    dataloader,
    tokenizer,
    label_expansions,
    device='cuda'
    
):
    """
    多任务评估
    当前版本：
    - 使用 verbalizer 做类别映射
    - 不再调用 model.predict()
    """
    model.eval()

    # ===== verbalizer =====

    intent_verbalizer = Verbalizer(
        tokenizer,
        label_expansions['intent']
    )

    section_verbalizer = Verbalizer(
        tokenizer,
        label_expansions['section']
    )
    worthiness_verbalizer = Verbalizer(
        tokenizer,
        label_expansions['worthiness']
    )
    # ===== collect =====

    all_preds_intent = []
    all_labels_intent = []
    all_preds_worthiness = []
    all_labels_worthiness = []
    all_preds_section = []
    all_labels_section = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            token_type_ids = batch['token_type_ids'].to(device)

            # ===== forward =====

            outputs = model(
                input_ids,
                attention_mask,
                token_type_ids
            )

            # ==================================================
            # intent
            # ==================================================

            intent_logits = intent_verbalizer.project(
                outputs['intent']
            )

            preds_intent = torch.argmax(
                intent_logits,
                dim=1
            )

            # ==================================================
            # section
            # ==================================================

            section_logits = section_verbalizer.project(
                outputs['section']
            )

            preds_section = torch.argmax(
                section_logits,
                dim=1
            )
            # ==================================================
            # worthiness
            # ==================================================

            worthiness_logits = worthiness_verbalizer.project(
                outputs['worthiness']
            )

            preds_worthiness = torch.argmax(
                worthiness_logits,
                dim=1
            )

            # ==================================================
            # collect
            # ==================================================

            all_preds_intent.extend(
                preds_intent.cpu().tolist()
            )

            all_labels_intent.extend(
                batch['intent_label'].cpu().tolist()
            )

            all_preds_section.extend(
                preds_section.cpu().tolist()
            )

            all_labels_section.extend(
                batch['section_label'].cpu().tolist()
            )
            all_preds_worthiness.extend(
                preds_worthiness.cpu().tolist()
            )

            all_labels_worthiness.extend(
                batch['worthiness_label'].long().cpu().tolist()
            )
    # ==========================================================
    # filter invalid labels
    # ==========================================================

    # ===== intent =====

    intent_mask = [
        label != -1
        for label in all_labels_intent
    ]

    intent_labels_filtered = [
        all_labels_intent[i]
        for i in range(len(all_labels_intent))
        if intent_mask[i]
    ]

    intent_preds_filtered = [
        all_preds_intent[i]
        for i in range(len(all_preds_intent))
        if intent_mask[i]
    ]

    # ===== section =====

    section_mask = [
        label != -1
        for label in all_labels_section
    ]

    section_labels_filtered = [
        all_labels_section[i]
        for i in range(len(all_labels_section))
        if section_mask[i]
    ]

    section_preds_filtered = [
        all_preds_section[i]
        for i in range(len(all_preds_section))
        if section_mask[i]
    ]
    worthiness_labels_filtered = all_labels_worthiness

    worthiness_preds_filtered = all_preds_worthiness
    # ==========================================================
    # metrics
    # ==========================================================

    metrics = {

        'intent': {

            'accuracy': accuracy_score(
                intent_labels_filtered,
                intent_preds_filtered
            ),

            'precision_macro': precision_score(
                intent_labels_filtered,
                intent_preds_filtered,
                average='macro',
                zero_division=0
            ),

            'recall_macro': recall_score(
                intent_labels_filtered,
                intent_preds_filtered,
                average='macro',
                zero_division=0
            ),

            'f1_macro': f1_score(
                intent_labels_filtered,
                intent_preds_filtered,
                average='macro',
                zero_division=0
            )
        },

        'section': {

            'accuracy': (
                accuracy_score(
                    section_labels_filtered,
                    section_preds_filtered
                )
                if len(section_labels_filtered) > 0
                else 0.0
            ),

            'precision_macro': (
                precision_score(
                    section_labels_filtered,
                    section_preds_filtered,
                    average='macro',
                    zero_division=0
                )
                if len(section_labels_filtered) > 0
                else 0.0
            ),

            'recall_macro': (
                recall_score(
                    section_labels_filtered,
                    section_preds_filtered,
                    average='macro',
                    zero_division=0
                )
                if len(section_labels_filtered) > 0
                else 0.0
            ),

            'f1_macro': (
                f1_score(
                    section_labels_filtered,
                    section_preds_filtered,
                    average='macro',
                    zero_division=0
                )
                if len(section_labels_filtered) > 0
                else 0.0
            )
        },
        'worthiness': {

            'accuracy': 
                accuracy_score(
                worthiness_labels_filtered,
                worthiness_preds_filtered
    ),

    'precision_macro': precision_score(
        worthiness_labels_filtered,
        worthiness_preds_filtered,
        average='macro',
        zero_division=0
    ),

    'recall_macro': recall_score(
        worthiness_labels_filtered,
        worthiness_preds_filtered,
        average='macro',
        zero_division=0
    ),

    'f1_macro': f1_score(
        worthiness_labels_filtered,
        worthiness_preds_filtered,
        average='macro',
        zero_division=0
    )
        }
    }

    return metrics


def save_model(
    model,
    optimizer,
    epoch,
    val_accuracy,
    save_path
):
    """保存模型检查点"""

    checkpoint = {

        'model_state_dict': model.state_dict(),

        'optimizer_state_dict': optimizer.state_dict(),

        'epoch': epoch,

        'val_accuracy': val_accuracy
    }

    torch.save(checkpoint, save_path)


def load_model(
    model,
    optimizer,
    load_path,
    device='cuda'
):
    """加载模型检查点"""

    checkpoint = torch.load(
        load_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

    epoch = checkpoint['epoch']

    val_accuracy = checkpoint['val_accuracy']

    return model, optimizer, epoch, val_accuracy


def save_metrics(metrics, save_path):
    """保存评估指标"""

    with open(
        save_path,
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False
        )


def print_metrics(
    metrics,
    epoch=None,
    prefix=''
):
    """打印评估指标"""

    if epoch is not None:

        print(f"\n{'=' * 50}")

        print(f"{prefix} Epoch {epoch} Metrics")

        print('=' * 50)

    for task, task_metrics in metrics.items():

        print(f"\n{task.capitalize()}:")

        for metric, value in task_metrics.items():

            print(f"  {metric}: {value:.4f}")


def count_parameters(model):
    """统计模型参数"""

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {

        'total': total_params,

        'trainable': trainable_params,

        'frozen': total_params - trainable_params
    }