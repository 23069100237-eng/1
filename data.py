#data.py
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LabelExpansionDict:
    """
    标签扩展字典 - 对应论文表1/表2的核心词映射
    
    引用意图识别 (citation_intent) 主任务标签：
    - Background: 背景介绍
    - Method: 方法引用
    - Result: 结果对比
    - Motivation: 动机说明
    - Future: 未来工作
    
    引文章节识别 (citation_section) 辅助任务标签：
    - Introduction: 引言
    - Related Work: 相关工作
    - Methods: 方法
    - Results: 结果
    - Discussion: 讨论
    
    引文价值识别 (citation_worthiness) 辅助任务标签：
    - Worthy: 有价值
    - Not Worthy: 无价值
    """
    
    @staticmethod
    def get_intent_expansion():
        """
        引用意图标签扩展映射 - 对应论文表1核心词
        推理时对同一意图的所有扩展词概率求平均（公式3）
        """
        return {
            'Background': [
                'background', 'context', 'previous', 'earlier', 'history',
                'foundation', 'basis', 'setting', 'introduction', 'prior'
            ],
            'Method': [
                'method', 'approach', 'technique', 'algorithm', 'procedure',
                'methodology', 'framework', 'tool', 'strategy', 'process'
            ],
            'Result': [
                'result', 'finding', 'outcome', 'conclusion', 'evidence',
                'observation', 'data', 'discovery', 'performance', 'comparison'
            ],
            'Motivation': [
                'motivation', 'purpose', 'objective', 'goal', 'aim',
                'reason', 'rationale', 'need', 'requirement', 'challenge'
            ],
            'Future': [
                'future', 'potential', 'direction', 'perspective', 'opportunity',
                'limitation', 'open', 'remaining', 'further', 'next'
            ]
        }
    
    @staticmethod
    def get_section_expansion():
        """
        引文章节标签扩展映射 - 对应论文表2核心词
        """
        return {
            'Introduction': [
                'introduction', 'intro', 'background', 'overview', 'motivation',
                'problem', 'objective', 'goal', 'purpose', 'aim'
            ],
            'Related Work': [
                'related', 'literature', 'review', 'previous', 'prior',
                'earlier', 'existing', 'state-of-the-art', 'survey', 'background'
            ],
            'Methods': [
                'method', 'methodology', 'approach', 'technique', 'algorithm',
                'procedure', 'experimental', 'setup', 'implementation', 'design'
            ],
            'Results': [
                'result', 'finding', 'outcome', 'evaluation', 'experiment',
                'performance', 'analysis', 'comparison', 'data', 'evidence'
            ],
            'Discussion': [
                'discussion', 'analysis', 'interpretation', 'implication', 'limitation',
                'conclusion', 'summary', 'insight', 'finding', 'observation'
            ]
        }
    
    @staticmethod
    def get_worthiness_expansion():
        """
        引文价值标签扩展映射
        """
        return {
            'Worthy': [
                'worthy', 'important', 'significant', 'valuable', 'crucial',
                'key', 'essential', 'meaningful', 'relevant', 'useful'
            ],
            'Not Worthy': [
                'unworthy', 'insignificant', 'unimportant', 'irrelevant', 'minor',
                'marginal', 'negligible', 'trivial', 'unnecessary', 'redundant'
            ]
        }

class CitationDataset(Dataset):
    """
    引文数据集类 - 支持ACL-ARC和原始SciCite数据集格式
    """
    
    def __init__(self, data_path, tokenizer, max_len=512, dataset_type='acl-arc'):
        """
        Args:
            data_path: 数据集文件路径（支持JSON和JSONL格式）
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
            dataset_type: 'acl-arc' 或 'scicite'（原始SciCite数据集）
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset_type = dataset_type
        
        # 标签到索引的映射
        self.intent_labels = ['Background', 'Method', 'Result', 'Motivation', 'Future']
        self.section_labels = ['Introduction', 'Related Work', 'Methods', 'Results', 'Discussion']
        self.intent_label2id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.section_label2id = {label: idx for idx, label in enumerate(self.section_labels)}
        
        # SciCite原始标签映射（小写转大写）
        self.scicite_intent_map = {
            'background': 'Background',
            'method': 'Method',
            'result': 'Result',
            'motivation': 'Motivation',
            'future': 'Future'
        }
        
        # SciCite章节名称标准化映射
        self.scicite_section_map = {
            'introduction': 'Introduction',
            'intro': 'Introduction',
            'background': 'Introduction',
            'methods': 'Methods',
            'method': 'Methods',
            'experimental': 'Methods',
            'results': 'Results',
            'result': 'Results',
            'discussion': 'Discussion',
            'conclusion': 'Discussion',
            'related work': 'Related Work',
            'related': 'Related Work',
            'literature review': 'Related Work',
            'discussion/conclusion': 'Discussion'
        }
    
    def _load_data(self, data_path):
        """加载数据集（支持JSON和JSONL格式）"""
        data = []
        
        # 判断文件格式
        with open(data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                # JSON数组格式
                f.seek(0)
                data = json.load(f)
            elif first_line.startswith('{'):
                # JSONL格式（每行一个JSON对象）
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        
        return data
    
    def _normalize_section(self, section_name):
        """标准化章节名称"""
        # 处理非字符串类型（如 NaN）
        if not section_name or isinstance(section_name, float):
            return ''
        
        # 转为小写并去除空格
        normalized = str(section_name).strip().lower()
        
        # 尝试映射
        if normalized in self.scicite_section_map:
            return self.scicite_section_map[normalized]
        
        # 如果是下划线或空格分隔，尝试分割处理
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        
        # 去除数字前缀（如 "4. discussion" -> "discussion"）
        words = normalized.split()
        if words and words[0].replace('.', '').isdigit():
            normalized = ' '.join(words[1:])
        
        # 检查去除数字后的名称
        if normalized in self.scicite_section_map:
            return self.scicite_section_map[normalized]
        
        # 首字母大写
        words = normalized.split()
        if words:
            capitalized = ' '.join(word.capitalize() for word in words)
            if capitalized in self.section_label2id:
                return capitalized
        
        return ''
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.data[idx]
        
        # 获取文本内容
        if self.dataset_type == 'acl-arc':
            text = item.get('text', '')
            text = text + " [MASK]."
            intent_label = item.get('intent', '')
            section_label = item.get('section', '')
            worthiness_label = item.get('worthiness', 0)
        else:  # 原始SciCite格式
            text = item.get('string', '')
            raw_label = item.get('label', '').lower()
            intent_label = self.scicite_intent_map.get(raw_label, '')
            section_label = self._normalize_section(item.get('sectionName', ''))
            # 使用isKeyCitation作为worthiness标签
            worthiness_label = 1 if item.get('isKeyCitation', False) else 0
        
        # 编码文本
        # ===== Prompt Learning Template =====
# [CLS] text [MASK] [SEP]

            prompt_text = text + " " + self.tokenizer.mask_token

            encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_len-10,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
)
        
        # 获取标签ID
        intent_id = self.intent_label2id.get(intent_label, -1)
        section_id = self.section_label2id.get(section_label, -1)
        worthiness = torch.tensor(worthiness_label, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'intent_label': torch.tensor(intent_id, dtype=torch.long),
            'section_label': torch.tensor(section_id, dtype=torch.long),
            'worthiness_label': worthiness
        }

def create_dataloader(dataset, batch_size=40, shuffle=True):
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

# 数据集样例格式说明
# ACL-ARC格式:
# {
#   "text": "The transformer architecture has revolutionized NLP...",
#   "intent": "Method",
#   "section": "Introduction",
#   "worthiness": 1
# }
#
# SciCite格式:
# {
#   "text": "Recent work has shown...",
#   "label": "Method",
#   "section": "Related Work",
#   "worthiness": 0
# }