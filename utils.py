"""
工具函数模块 - 包含评估指标、损失计算、日志记录等
"""
#utils.py
import torch
import json
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
    mask = labels != -1  # 忽略-1标签
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def evaluate_multitask(model, dataloader, tokenizer, label_expansions, device='cuda'):
    model.eval()
    
    all_preds_intent = []
    all_preds_section = []
    all_preds_worthiness = []
    all_labels_intent = []
    all_labels_section = []
    all_labels_worthiness = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # ===== intent =====
            preds_intent, _ = model.predict(
                outputs['intent'],
                label_expansions['intent'],
                tokenizer
            )
            
            # ===== section =====
            preds_section, _ = model.predict(
                outputs['section'],
                label_expansions['section'],
                tokenizer
            )
            
            # ===== worthiness（关键修复）=====
            # 保证 shape 一定是 (batch_size,)
            preds_worthiness = (torch.sigmoid(outputs['worthiness']) > 0.5).int().view(-1)
            
            # ===== 收集 =====
            all_preds_intent.extend(preds_intent.cpu().numpy().tolist())
            all_preds_section.extend(preds_section.cpu().numpy().tolist())
            all_preds_worthiness.extend(preds_worthiness.cpu().numpy().tolist())
            
            all_labels_intent.extend(batch['intent_label'].cpu().numpy().tolist())
            all_labels_section.extend(batch['section_label'].cpu().numpy().tolist())
            all_labels_worthiness.extend(batch['worthiness_label'].cpu().numpy().tolist())
    
    # ===== 过滤 -1 =====
    intent_mask = [l != -1 for l in all_labels_intent]
    section_mask = [l != -1 for l in all_labels_section]
    
    # 提取过滤后的数据
    intent_labels_filtered = [all_labels_intent[i] for i in range(len(all_labels_intent)) if intent_mask[i]]
    intent_preds_filtered = [all_preds_intent[i] for i in range(len(all_preds_intent)) if intent_mask[i]]
    section_labels_filtered = [all_labels_section[i] for i in range(len(all_labels_section)) if section_mask[i]]
    section_preds_filtered = [all_preds_section[i] for i in range(len(all_preds_section)) if section_mask[i]]
    
    # 计算多分类任务的AUC（使用one-vs-rest策略）
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    # 意图识别指标
    intent_labels_bin = label_binarize(intent_labels_filtered, classes=[0, 1, 2, 3, 4])
    intent_auc = roc_auc_score(intent_labels_bin, np.eye(5)[intent_preds_filtered], average='macro', multi_class='ovr')
    
    # 章节识别指标
    section_labels_bin = label_binarize(section_labels_filtered, classes=[0, 1, 2, 3, 4])
    section_auc = roc_auc_score(section_labels_bin, np.eye(5)[section_preds_filtered], average='macro', multi_class='ovr')
    
    metrics = {
        'intent': {
            'accuracy': accuracy_score(intent_labels_filtered, intent_preds_filtered),
            'precision_macro': precision_score(intent_labels_filtered, intent_preds_filtered, average='macro'),
            'recall_macro': recall_score(intent_labels_filtered, intent_preds_filtered, average='macro'),
            'f1_macro': f1_score(intent_labels_filtered, intent_preds_filtered, average='macro'),
            'auc_macro': intent_auc
        },
        'section': {
            'accuracy': accuracy_score(section_labels_filtered, section_preds_filtered),
            'precision_macro': precision_score(section_labels_filtered, section_preds_filtered, average='macro'),
            'recall_macro': recall_score(section_labels_filtered, section_preds_filtered, average='macro'),
            'f1_macro': f1_score(section_labels_filtered, section_preds_filtered, average='macro'),
            'auc_macro': section_auc
        },
        'worthiness': {
            'accuracy': accuracy_score(all_labels_worthiness, all_preds_worthiness),
            'precision_macro': precision_score(all_labels_worthiness, all_preds_worthiness, average='macro'),
            'recall_macro': recall_score(all_labels_worthiness, all_preds_worthiness, average='macro'),
            'f1_macro': f1_score(all_labels_worthiness, all_preds_worthiness, average='macro'),
            'auc_macro': roc_auc_score(all_labels_worthiness, all_preds_worthiness)
        }
    }
    
    return metrics

def save_model(model, optimizer, epoch, val_accuracy, save_path):
    """保存模型检查点"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, save_path)

def load_model(model, optimizer, load_path, device='cuda'):
    """加载模型检查点"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_accuracy = checkpoint['val_accuracy']
    return model, optimizer, epoch, val_accuracy

def save_metrics(metrics, save_path):
    """保存评估指标"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

def print_metrics(metrics, epoch=None, prefix=''):
    """打印评估指标"""
    if epoch is not None:
        print(f"\n{'='*50}")
        print(f"{prefix} Epoch {epoch} Metrics")
        print('='*50)
    
    for task, task_metrics in metrics.items():
        print(f"\n{task.capitalize()}:")
        for metric, value in task_metrics.items():
            print(f"  {metric}: {value:.4f}")

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }