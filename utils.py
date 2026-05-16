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
    device
):

    from verbalizer import Verbalizer

    model.eval()

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

    all_intent_preds = []
    all_intent_labels = []

    all_section_preds = []
    all_section_labels = []

    all_worthiness_preds = []
    all_worthiness_labels = []

    with torch.no_grad():

        for batch in dataloader:

            # ==================================================
            # intent
            # ==================================================

            intent_input_ids = (
                batch['intent_input_ids']
                .to(device)
            )

            intent_attention_mask = (
                batch['intent_attention_mask']
                .to(device)
            )

            intent_token_type_ids = (
                batch['intent_token_type_ids']
                .to(device)
            )

            # ==================================================
            # section
            # ==================================================

            section_input_ids = (
                batch['section_input_ids']
                .to(device)
            )

            section_attention_mask = (
                batch['section_attention_mask']
                .to(device)
            )

            section_token_type_ids = (
                batch['section_token_type_ids']
                .to(device)
            )

            # ==================================================
            # worthiness
            # ==================================================

            worthiness_input_ids = (
                batch['worthiness_input_ids']
                .to(device)
            )

            worthiness_attention_mask = (
                batch['worthiness_attention_mask']
                .to(device)
            )

            worthiness_token_type_ids = (
                batch['worthiness_token_type_ids']
                .to(device)
            )

            # ==================================================
            # labels
            # ==================================================

            labels_intent = (
                batch['intent_label']
                .to(device)
            )

            labels_section = (
                batch['section_label']
                .to(device)
            )

            labels_worthiness = (
                batch['worthiness_label']
                .long()
                .to(device)
            )

            # ==================================================
            # forward
            # ==================================================

            intent_outputs = model.forward_single_task(

                intent_input_ids,

                intent_attention_mask,

                intent_token_type_ids,

                model.prompt_mlp_intent
            )

            section_outputs = model.forward_single_task(

                section_input_ids,

                section_attention_mask,

                section_token_type_ids,

                model.prompt_mlp_section
            )

            worthiness_outputs = model.forward_single_task(

                worthiness_input_ids,

                worthiness_attention_mask,

                worthiness_token_type_ids,

                model.prompt_mlp_worthiness
            )

            # ==================================================
            # verbalizer
            # ==================================================

            intent_logits = intent_verbalizer.project(
                intent_outputs
            )

            section_logits = section_verbalizer.project(
                section_outputs
            )

            worthiness_logits = worthiness_verbalizer.project(
                worthiness_outputs
            )

            # ==================================================
            # predictions
            # ==================================================

            intent_preds = torch.argmax(
                intent_logits,
                dim=1
            )

            section_preds = torch.argmax(
                section_logits,
                dim=1
            )

            worthiness_preds = torch.argmax(
                worthiness_logits,
                dim=1
            )

            # ==================================================
            # save
            # ==================================================

            valid_intent = labels_intent != -1

            all_intent_preds.extend(
                intent_preds[valid_intent]
                .cpu()
                .tolist()
            )

            all_intent_labels.extend(
                labels_intent[valid_intent]
                .cpu()
                .tolist()
            )

            valid_section = labels_section != -1

            all_section_preds.extend(
                section_preds[valid_section]
                .cpu()
                .tolist()
            )

            all_section_labels.extend(
                labels_section[valid_section]
                .cpu()
                .tolist()
            )

            all_worthiness_preds.extend(
                worthiness_preds
                .cpu()
                .tolist()
            )

            all_worthiness_labels.extend(
                labels_worthiness
                .cpu()
                .tolist()
            )

    # ==================================================
    # metrics
    # ==================================================

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support
    )

    metrics = {}

    # ================= intent =================

    intent_precision, intent_recall, intent_f1, _ = (
        precision_recall_fscore_support(
            all_intent_labels,
            all_intent_preds,
            average='macro',
            zero_division=0
        )
    )

    metrics['intent'] = {

        'accuracy': accuracy_score(
            all_intent_labels,
            all_intent_preds
        ),

        'precision': intent_precision,

        'recall': intent_recall,

        'f1': intent_f1
    }

    # ================= section =================

    if len(all_section_labels) > 0:

        section_precision, section_recall, section_f1, _ = (
            precision_recall_fscore_support(
                all_section_labels,
                all_section_preds,
                average='macro',
                zero_division=0
            )
        )

        metrics['section'] = {

            'accuracy': accuracy_score(
                all_section_labels,
                all_section_preds
            ),

            'precision': section_precision,

            'recall': section_recall,

            'f1': section_f1
        }

    else:

        metrics['section'] = {

            'accuracy': 0.0,

            'precision': 0.0,

            'recall': 0.0,

            'f1': 0.0
        }

    # ================= worthiness =================

    worthiness_precision, worthiness_recall, worthiness_f1, _ = (
        precision_recall_fscore_support(
            all_worthiness_labels,
            all_worthiness_preds,
            average='macro',
            zero_division=0
        )
    )

    metrics['worthiness'] = {

        'accuracy': accuracy_score(
            all_worthiness_labels,
            all_worthiness_preds
        ),

        'precision': worthiness_precision,

        'recall': worthiness_recall,

        'f1': worthiness_f1
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