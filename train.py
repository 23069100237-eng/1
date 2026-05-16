# train.py

import argparse
import os
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from tqdm import tqdm

from data import CitationDataset, create_dataloader, LabelExpansionDict
from model import CitationPromptModel
from verbalizer import Verbalizer

from config import *

from utils import (
    setup_logging,
    evaluate_multitask,
    save_model,
    load_model,
    print_metrics,
    count_parameters
)


def parse_args():

    parser = argparse.ArgumentParser(
        description='Prompt Learning Citation Intent Classification'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='scicite',
        choices=['acl-arc', 'scicite']
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        '--max_len',
        type=int,
        default=MAX_LEN
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=EARLY_STOPPING_PATIENCE
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda']
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR
    )

    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        default=USE_MIXED_PRECISION
    )

    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(
        args.output_dir,
        'training.log'
    )

    logger = setup_logging(log_file)

    logger.info(f"开始训练: {args.dataset}")

    if args.device == 'cuda' and not torch.cuda.is_available():

        logger.warning("CUDA不可用，切换CPU")

        args.device = 'cpu'

    device = torch.device(args.device)

    # =========================
    # tokenizer
    # =========================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    # =========================
    # label expansion
    # =========================

    label_expansions = {
        'intent': LabelExpansionDict.get_intent_expansion(),
        'section': LabelExpansionDict.get_section_expansion(),
        'worthiness': LabelExpansionDict.get_worthiness_expansion()
    }

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

    # =========================
    # dataset
    # =========================

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

    train_dataloader = create_dataloader(
        train_dataset,
        args.batch_size,
        True
    )

    val_dataloader = create_dataloader(
        val_dataset,
        args.batch_size,
        False
    )

    logger.info(f"train size: {len(train_dataset)}")
    logger.info(f"val size: {len(val_dataset)}")

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
    ).to(device)

    # =========================
    # params
    # =========================

    params = count_parameters(model)

    logger.info(f"总参数: {params['total']:,}")
    logger.info(f"可训练参数: {params['trainable']:,}")

    # =========================
    # optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=args.lr
    )

    # =========================
    # amp
    # =========================

    use_amp = (
        args.mixed_precision
        and
        device.type == 'cuda'
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp
    )

    best_val_accuracy = 0.0

    patience_counter = 0

    save_path = os.path.join(
        args.output_dir,
        'best_model.pt'
    )

    logger.info("开始训练")

    # =====================================================
    # train loop
    # =====================================================

    for epoch in range(args.epochs):

        model.train()

        epoch_loss = 0.0

        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for batch in progress_bar:

            # ==================================================
            # intent inputs
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
            # section inputs
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
            # worthiness inputs
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

            labels_intent = batch['intent_label'].to(device)

            labels_section = batch['section_label'].to(device)

            labels_worthiness = (
                batch['worthiness_label']
                .long()
                .to(device)
            )

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                outputs = {

                    'intent':

                        model.forward_single_task(

                            intent_input_ids,

                            intent_attention_mask,

                            intent_token_type_ids,

                            model.prompt_mlp_intent
                        ),

                    'section':

                        model.forward_single_task(

                            section_input_ids,

                            section_attention_mask,

                            section_token_type_ids,

                            model.prompt_mlp_section
                        ),

                    'worthiness':

                        model.forward_single_task(

                            worthiness_input_ids,

                            worthiness_attention_mask,

                            worthiness_token_type_ids,

                            model.prompt_mlp_worthiness
                        )
                }

                # ======================
                # verbalizer
                # ======================

                intent_logits = intent_verbalizer.project(
                    outputs['intent']
                )

                section_logits = section_verbalizer.project(
                    outputs['section']
                )

                worthiness_logits = worthiness_verbalizer.project(
                    outputs['worthiness']
                )

                # ======================
                # task losses
                # ======================

                intent_loss = F.cross_entropy(
                    intent_logits,
                    labels_intent,
                    ignore_index=-1
                )

                section_loss = F.cross_entropy(
                    section_logits,
                    labels_section,
                    ignore_index=-1
                )

                worthiness_loss = F.cross_entropy(
                    worthiness_logits,
                    labels_worthiness
                )

                # ======================
                # multitask loss
                # ======================

                if args.dataset == 'scicite':

                    loss = (
                        1.0 * intent_loss
                        +
                        0.5 * worthiness_loss
                    )

                else:

                    loss = (
                        1.0 * intent_loss
                        +
                        0.5 * section_loss
                        +
                        0.5 * worthiness_loss
                    )

                # ======================
                # soft parameter sharing
                # ======================

                soft_loss = model.compute_soft_sharing_loss()

                loss += 1e-5 * soft_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            epoch_loss += loss.item()

            num_batches += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

        avg_loss = epoch_loss / num_batches

        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}"
        )

        logger.info("开始验证")

        val_metrics = evaluate_multitask(
            model=model,
            dataloader=val_dataloader,
            tokenizer=tokenizer,
            label_expansions=label_expansions,
            device=device
        )

        print_metrics(
            val_metrics,
            epoch=epoch + 1,
            prefix='验证'
        )

        if args.dataset == 'scicite':

            current_val_accuracy = (
                val_metrics['intent']['accuracy']
                +
                val_metrics['worthiness']['accuracy']
            ) / 2

        else:

            current_val_accuracy = (
                val_metrics['intent']['accuracy']
                +
                val_metrics['section']['accuracy']
                +
                val_metrics['worthiness']['accuracy']
            ) / 3

        logger.info(
            f"Val Accuracy: {current_val_accuracy:.4f}"
        )

        if current_val_accuracy > best_val_accuracy:

            best_val_accuracy = current_val_accuracy

            patience_counter = 0

            logger.info(
                f"发现更优模型: {best_val_accuracy:.4f}"
            )

            save_model(
                model,
                optimizer,
                epoch + 1,
                best_val_accuracy,
                save_path
            )

        else:

            patience_counter += 1

            logger.info(
                f"EarlyStopping: "
                f"{patience_counter}/{args.patience}"
            )

            if patience_counter >= args.patience:

                logger.info("触发早停")

                break

    logger.info(
        f"训练结束，最佳准确率: "
        f"{best_val_accuracy:.4f}"
    )


def inference():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        required=True
    )

    parser.add_argument(
        '--text',
        type=str,
        required=True
    )

    parser.add_argument(
        '--task',
        type=str,
        default='intent',
        choices=[
            'intent',
            'section',
            'worthiness'
        ]
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model = CitationPromptModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        prompt_length=PROMPT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        alpha=L2_ALPHA
    )

    model, _, _, _ = load_model(
        model,
        None,
        args.model_path,
        device
    )

    model = model.to(device)

    model.eval()

    label_expansions = {

        'intent':
            LabelExpansionDict.get_intent_expansion(),

        'section':
            LabelExpansionDict.get_section_expansion(),

        'worthiness':
            LabelExpansionDict.get_worthiness_expansion()
    }

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

    # ==================================================
    # task-specific prompt
    # ==================================================

    if args.task == 'intent':

        prompt_text = (
            f"{args.text} "
            f"This citation expresses "
            f"{tokenizer.mask_token}."
        )

    elif args.task == 'section':

        prompt_text = (
            f"{args.text} "
            f"This citation appears in the "
            f"{tokenizer.mask_token} section."
        )

    else:

        prompt_text = (
            f"{args.text} "
            f"This citation is "
            f"{tokenizer.mask_token}."
        )

    encoding = tokenizer(

        prompt_text,

        truncation=True,

        max_length=MAX_LEN,

        padding='max_length',

        return_attention_mask=True,

        return_token_type_ids=True,

        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)

    attention_mask = encoding['attention_mask'].to(device)

    token_type_ids = encoding['token_type_ids'].to(device)

    with torch.no_grad():

        if args.task == 'intent':

            mask_logits = model.forward_single_task(

                input_ids,

                attention_mask,

                token_type_ids,

                model.prompt_mlp_intent
            )

            logits = intent_verbalizer.project(
                mask_logits
            )

            labels = list(
                label_expansions['intent'].keys()
            )

        elif args.task == 'section':

            mask_logits = model.forward_single_task(

                input_ids,

                attention_mask,

                token_type_ids,

                model.prompt_mlp_section
            )

            logits = section_verbalizer.project(
                mask_logits
            )

            labels = list(
                label_expansions['section'].keys()
            )

        else:

            mask_logits = model.forward_single_task(

                input_ids,

                attention_mask,

                token_type_ids,

                model.prompt_mlp_worthiness
            )

            logits = worthiness_verbalizer.project(
                mask_logits
            )

            labels = list(
                label_expansions['worthiness'].keys()
            )

        predictions = torch.argmax(
            logits,
            dim=1
        )

    print("\n预测结果:")

    print(labels[predictions.item()])
    
if __name__ == '__main__':

    import sys

    if (
        len(sys.argv) > 1
        and
        sys.argv[1] == 'inference'
    ):

        inference()

    else:

        main()