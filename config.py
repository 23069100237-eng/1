"""
配置文件 - 包含数据集路径、超参数配置

论文损失权重配置：
- ACL-ARC: λ_int=1.0, λ_wor=0.32, λ_sec=0.16
- SciCite: λ_int=1.0, λ_wor=0.35, λ_sec=0.0
"""
import os

# 基础模型配置
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
MODEL_DIR = os.path.abspath('./scibert_scivocab_uncased')  # 本地模型目录
HIDDEN_SIZE = 768
PROMPT_LENGTH = 10  # 使用10个[unused] token
DROPOUT_RATE = 0.3
L2_ALPHA = 1e-5  # L2正则化系数

# 训练配置（根据论文实验参数）
BATCH_SIZE = 40  # 论文中批量大小为40
MAX_LEN = 512  # 论文中最大序列长度为512
LEARNING_RATE = 1e-4  # 论文中学习率为0.0001
NUM_EPOCHS = 10  # 论文中训练轮次为10轮
EARLY_STOPPING_PATIENCE = 5  # 论文中早停条件为连续5轮验证集性能无提升

# 损失权重配置
LOSS_WEIGHTS = {
    'acl-arc': {
        'lambda_int': 1.0,
        'lambda_sec': 0.16,
        'lambda_wor': 0.32
    },
    'scicite': {
        'lambda_int': 1.0,
        'lambda_sec': 0.0,
        'lambda_wor': 0.35
    }
}

# 数据集路径配置
DATA_DIR = './scicite'
OUTPUT_DIR = './output'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集文件名
DATA_FILES = {
    'acl-arc': {
        'train': os.path.join(DATA_DIR, 'acl-arc_train.json'),
        'val': os.path.join(DATA_DIR, 'acl-arc_val.json'),
        'test': os.path.join(DATA_DIR, 'acl-arc_test.json')
    },
    'scicite': {
        'train': os.path.join(DATA_DIR, 'train.jsonl'),
        'val': os.path.join(DATA_DIR, 'dev.jsonl'),
        'test': os.path.join(DATA_DIR, 'test.jsonl')
    }
}

# 模型保存路径
MODEL_CHECKPOINT = os.path.join(OUTPUT_DIR, 'best_model.pt')
MODEL_CONFIG = os.path.join(OUTPUT_DIR, 'model_config.json')

# 日志配置
LOG_FILE = os.path.join(OUTPUT_DIR, 'training.log')

# 设备配置
DEVICE = 'cuda' if __name__ == '__main__' else 'cpu'  # 延迟初始化

# 混合精度训练配置
USE_MIXED_PRECISION = True