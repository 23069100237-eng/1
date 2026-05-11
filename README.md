# 基于提示学习与多任务学习的学术文献引用意图识别研究

复现论文《基于提示学习与多任务学习的学术文献引用意图识别研究》的核心框架。

## 代码结构

```
./
├── data.py          # 数据预处理模块（数据集类、标签扩展字典）
├── model.py         # 模型定义（PromptMLP、TaskHead、完整模型架构）
├── config.py        # 配置文件（超参数、数据集路径）
├── utils.py         # 工具函数（评估指标、模型保存/加载）
├── train.py         # 训练脚本（训练循环、验证循环、早停机制）
├── data/            # 数据集目录
│   ├── acl-arc_train.json
│   ├── acl-arc_val.json
│   ├── acl-arc_test.json
│   ├── scicite_train.json
│   ├── scicite_val.json
│   └── scicite_test.json
└── output/          # 输出目录（模型检查点、日志）
```

## 环境依赖

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
pip install tqdm
```

## 数据集格式

支持两种数据集格式：

### ACL-ARC格式
```json
{
  "text": "The transformer architecture has revolutionized NLP...",
  "intent": "Method",
  "section": "Introduction",
  "worthiness": 1
}
```

### SciCite格式
```json
{
  "text": "Recent work has shown...",
  "label": "Method",
  "section": "Related Work",
  "worthiness": 0
}
```

## 标签定义

### 引用意图识别（主任务）
- Background: 背景介绍
- Method: 方法引用
- Result: 结果对比
- Motivation: 动机说明
- Future: 未来工作

### 引文章节识别（辅助任务）
- Introduction: 引言
- Related Work: 相关工作
- Methods: 方法
- Results: 结果
- Discussion: 讨论

### 引文价值识别（辅助任务）
- Worthy (1): 有价值
- Not Worthy (0): 无价值

## 训练命令

### 在ACL-ARC数据集上训练

```bash
python train.py --dataset acl-arc --batch_size 40 --max_len 512 --lr 1e-4 \
    --epochs 50 --patience 5 --device cuda --mixed_precision
```

### 在SciCite数据集上训练

```bash
python train.py --dataset scicite --batch_size 40 --max_len 512 --lr 1e-4 \
    --epochs 50 --patience 5 --device cuda --mixed_precision
```

### 单卡RTX 3090运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset acl-arc --batch_size 40 \
    --max_len 512 --mixed_precision
```

## 推理命令

```bash
python train.py inference --model_path ./output/best_model.pt --text "Your citation text here"
```

## 损失权重配置

根据论文设置：

| 数据集 | λ_int | λ_sec | λ_wor |
|--------|-------|-------|-------|
| ACL-ARC | 1.0 | 0.16 | 0.32 |
| SciCite | 1.0 | 0.0 | 0.35 |

## 模型架构

1. **基础模型**: allenai/scibert_scivocab_uncased（参数冻结）
2. **P-tuning**: 10个[unused] token通过两层MLP映射为连续向量
3. **输入格式**: [连续提示向量] + [CLS] text [SEP] [MASK] [SEP]
4. **预测头**: 三个独立的线性层，分别对应三个任务

## 损失函数（公式6）

```
Loss_total = λ_int*L_int + λ_sec*L_sec + λ_wor*L_wor + α*(||θ_int||² + ||θ_sec||² + ||θ_wor||²)
```

其中 θ 仅包含三个任务的PromptMLP和TaskHead参数。

## 标签扩展推理

- 公式3：对同一意图的所有扩展词概率求平均
- 公式4：取最高分作为预测

## 训练配置

- 优化器: Adam, lr=1e-4
- Batch size: 40
- Max length: 512
- Dropout: 0.3
- Early stopping: patience=5 on validation accuracy