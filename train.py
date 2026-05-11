"""
训练脚本 - 包含完整的训练循环、验证循环、早停机制、混合精度训练

复现论文《基于提示学习与多任务学习的学术文献引用意图识别研究》的核心框架
"""

import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from data import CitationDataset, create_dataloader, LabelExpansionDict
from model import CitationPromptModel
from config import *
from utils import (
    setup_logging, evaluate_multitask, save_model, load_model,
    print_metrics, count_parameters
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多任务提示学习引文意图识别训练')

    parser.add_argument('--dataset', type=str, default='acl-arc',
                        choices=['acl-arc', 'scicite'],
                        help='数据集类型')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--max_len', type=int, default=MAX_LEN,
                        help='最大序列长度')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='早停耐心值')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='训练设备')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--mixed_precision', action='store_true', default=USE_MIXED_PRECISION,
                        help='是否使用混合精度训练')

    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'training.log')
    logger = setup_logging(log_file)

    logger.info(f"开始训练，数据集: {args.dataset}")

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，将使用CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    # tokenizer - 从本地加载
    logger.info(f"加载tokenizer: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    # 标签
    label_expansions = {
        'intent': LabelExpansionDict.get_intent_expansion(),
        'section': LabelExpansionDict.get_section_expansion(),
        'worthiness': LabelExpansionDict.get_worthiness_expansion()
    }

    # 数据集
    logger.info(f"加载{args.dataset}数据集")
    train_dataset = CitationDataset(
        DATA_FILES[args.dataset]['train'],
        tokenizer,
        args.max_len,
        args.dataset
    )
    val_dataset = CitationDataset(
        DATA_FILES[args.dataset]['val'],
        tokenizer,
        args.max_len,
        args.dataset
    )

    print("train_dataset size:", len(train_dataset))
    print("val_dataset size:", len(val_dataset))

    train_dataloader = create_dataloader(train_dataset, args.batch_size, True)
    val_dataloader = create_dataloader(val_dataset, args.batch_size, False)

    print("train_dataloader size:", len(train_dataloader))
    print("val_dataloader size:", len(val_dataloader))

    # 模型 - 从本地加载
    logger.info(f"创建模型: {MODEL_DIR}")
    model = CitationPromptModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        prompt_length=PROMPT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        alpha=L2_ALPHA
    ).to(device)

    # 参数统计
    params = count_parameters(model)
    logger.info(f"模型参数总数: {params['total']:,}")
    logger.info(f"可训练参数: {params['trainable']:,}")
    logger.info(f"冻结参数: {params['frozen']:,}")

    # 优化器
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=args.lr)

    # 混合精度
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    loss_weights = LOSS_WEIGHTS[args.dataset]
    logger.info(f"损失权重: {loss_weights}")

    best_val_accuracy = 0
    patience_counter = 0
    save_path = os.path.join(args.output_dir, "best_model.pt")

    logger.info("开始训练")

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                labels_intent = batch['intent_label'].to(device)
                labels_section = batch['section_label'].to(device)
                labels_worthiness = batch['worthiness_label'].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(input_ids, attention_mask, token_type_ids)

                    losses = model.compute_loss(
                        outputs['intent'],
                        outputs['section'],
                        outputs['worthiness'],
                        labels_intent,
                        labels_section,
                        labels_worthiness,
                        **loss_weights
                    )

                scaler.scale(losses['total']).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += losses['total'].item()
                num_batches += 1

                progress_bar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'int': f"{losses['intent'].item():.4f}",
                    'sec': f"{losses['section'].item():.4f}",
                    'wor': f"{losses['worthiness'].item():.4f}"
                })
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error(f"GPU显存不足: {e}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                raise e

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1} - 训练损失: {avg_loss:.4f}")

        # 验证
        logger.info("开始验证")
        val_metrics = evaluate_multitask(
            model, val_dataloader, tokenizer, label_expansions, device
        )
        print_metrics(val_metrics, epoch=epoch+1, prefix='验证')
        logger.info(f"Epoch {epoch+1} 验证指标: {val_metrics}")

        # 早停
        current_val_accuracy = val_metrics['intent']['accuracy']
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            patience_counter = 0
            logger.info(f"验证准确率提升至 {best_val_accuracy:.4f}，保存模型")
            save_model(model, optimizer, epoch+1, best_val_accuracy, save_path)
        else:
            patience_counter += 1
            logger.info(f"验证准确率未提升，耐心计数: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                logger.info("早停触发，停止训练")
                break

    logger.info(f"训练完成，最佳验证准确率: {best_val_accuracy:.4f}")
    logger.info(f"模型已保存至: {save_path}")

def inference():
    """推理函数"""
    parser = argparse.ArgumentParser(description='推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--text', type=str, required=True,
                        help='待预测的引文文本')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # tokenizer - 从本地加载
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    # 创建模型 - 从本地加载
    model = CitationPromptModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        prompt_length=PROMPT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        alpha=L2_ALPHA
    )

    # 加载模型权重
    model, _, _, _ = load_model(model, None, args.model_path, args.device)
    model = model.to(args.device)
    model.eval()

    # 加载标签扩展字典
    label_expansions = {
        'intent': LabelExpansionDict.get_intent_expansion(),
        'section': LabelExpansionDict.get_section_expansion(),
        'worthiness': LabelExpansionDict.get_worthiness_expansion()
    }

    # 编码文本
    encoding = tokenizer(
        args.text,
        truncation=True,
        max_length=MAX_LEN - 3,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(args.device)
    attention_mask = encoding['attention_mask'].to(args.device)
    token_type_ids = encoding['token_type_ids'].to(args.device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)

        intent_preds, intent_probs = model.predict(
            outputs['intent'],
            label_expansions['intent'],
            tokenizer
        )

        section_preds, section_probs = model.predict(
            outputs['section'],
            label_expansions['section'],
            tokenizer
        )

        worthiness_pred = (outputs['worthiness'] > 0.5).int().item()

    intent_labels = list(label_expansions['intent'].keys())
    section_labels = list(label_expansions['section'].keys())

    print("\n推理结果:")
    print(f"引文意图: {intent_labels[intent_preds.item()]} (概率: {intent_probs.max().item():.4f})")
    print(f"引文章节: {section_labels[section_preds.item()]} (概率: {section_probs.max().item():.4f})")
    print(f"引文价值: {'有价值' if worthiness_pred == 1 else '无价值'}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'inference':
        inference()
    else:
        main()