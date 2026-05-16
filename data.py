# data.py

import os
import json
import random
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LabelExpansionDict:

    @staticmethod
    def get_intent_expansion():

        """
        ACL-ARC + SciCite
        统一 verbalizer label
        """

        return {

            # ===== SciCite =====

            'Background': [
                'background',
                'context',
                'previous',
                'prior',
                'foundation',
                'basis',
                'history',
                'overview'
            ],

            'Method': [
                'method',
                'approach',
                'technique',
                'algorithm',
                'procedure',
                'framework',
                'strategy',
                'tool'
            ],

            'Result': [
                'result',
                'finding',
                'outcome',
                'evidence',
                'observation',
                'performance',
                'comparison',
                'analysis'
            ],

            # ===== ACL-ARC =====

            'Compare/contrast': [
                'compare',
                'contrast',
                'different',
                'similar',
                'comparison',
                'competitive',
                'alternative',
                'better'
            ],

            'Extends': [
                'extend',
                'improve',
                'enhance',
                'expand',
                'build',
                'develop',
                'refine',
                'upgrade'
            ],

            'Future work': [
                'future',
                'direction',
                'potential',
                'further',
                'remaining',
                'next',
                'opportunity',
                'limitation'
            ],

            'Motivation': [
                'motivation',
                'purpose',
                'goal',
                'objective',
                'reason',
                'aim',
                'need',
                'challenge'
            ],

            'Uses': [
                'use',
                'utilize',
                'apply',
                'adopt',
                'employ',
                'integrate',
                'borrow',
                'reference'
            ]
        }

    @staticmethod
    def get_section_expansion():

        return {

            'Introduction': [
                'introduction',
                'overview',
                'background',
                'motivation',
                'objective',
                'purpose'
            ],

            'Related Work': [
                'related',
                'literature',
                'survey',
                'previous',
                'prior',
                'existing'
            ],

            'Methods': [
                'method',
                'approach',
                'algorithm',
                'procedure',
                'implementation',
                'experimental'
            ],

            'Results': [
                'result',
                'finding',
                'evaluation',
                'performance',
                'analysis',
                'comparison'
            ],

            'Discussion': [
                'discussion',
                'conclusion',
                'interpretation',
                'implication',
                'summary',
                'insight'
            ]
        }

    @staticmethod
    def get_worthiness_expansion():

        return {

            'Worthy': [
                'important',
                'significant',
                'valuable',
                'essential',
                'meaningful',
                'relevant',
                'useful'
            ],

            'Not Worthy': [
                'unimportant',
                'irrelevant',
                'minor',
                'trivial',
                'redundant',
                'insignificant'
            ]
        }


class CitationDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=512,
        dataset_type='acl-arc'
    ):

        self.data = self._load_data(data_path)

        self.tokenizer = tokenizer

        self.max_len = max_len

        self.dataset_type = dataset_type

        # ==================================================
        # label definitions
        # ==================================================

        if dataset_type == 'scicite':

            self.intent_labels = [
                'Background',
                'Method',
                'Result'
            ]

            self.section_labels = []

        else:

            self.intent_labels = [
                'Background',
                'Compare/contrast',
                'Extends',
                'Future work',
                'Motivation',
                'Uses'
            ]

            self.section_labels = [
                'Introduction',
                'Related Work',
                'Methods',
                'Results',
                'Discussion'
            ]

        self.intent_label2id = {
            label: idx
            for idx, label in enumerate(self.intent_labels)
        }

        self.section_label2id = {
            label: idx
            for idx, label in enumerate(self.section_labels)
        }

        # ==================================================
        # SciCite mapping
        # ==================================================

        self.scicite_intent_map = {

            'background': 'Background',
            'method': 'Method',
            'result': 'Result'
        }

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
            'literature review': 'Related Work'
        }

    def _load_data(self, data_path):

        data = []

        with open(data_path, 'r', encoding='utf-8') as f:

            first_line = f.readline().strip()

            if first_line.startswith('['):

                f.seek(0)

                data = json.load(f)

            elif first_line.startswith('{'):

                f.seek(0)

                for line in f:

                    line = line.strip()

                    if line:

                        data.append(json.loads(line))

            else:

                raise ValueError(
                    f"Unsupported format: {data_path}"
                )

        return data

    def _normalize_section(self, section_name):

        if not section_name:

            return ''

        normalized = str(section_name).strip().lower()

        normalized = (
            normalized
            .replace('_', ' ')
            .replace('-', ' ')
        )

        if normalized in self.scicite_section_map:

            return self.scicite_section_map[normalized]

        return ''

    def build_intent_prompt(self, text):

        return (
            f"{text} "
            f"This citation expresses "
            f"{self.tokenizer.mask_token}."
        )

    def build_section_prompt(self, text):

        return (
            f"{text} "
            f"This citation appears in the "
            f"{self.tokenizer.mask_token} section."
        )

    def build_worthiness_prompt(self, text):

        return (
            f"{text} "
            f"This citation is "
            f"{self.tokenizer.mask_token}."
        )

    def encode_text(self, text):

        encoding = self.tokenizer(

            text,

            truncation=True,

            max_length=self.max_len - 10,

            padding='max_length',

            return_attention_mask=True,

            return_token_type_ids=True,

            return_tensors='pt'
        )

        return {

            'input_ids':
                encoding['input_ids'].flatten(),

            'attention_mask':
                encoding['attention_mask'].flatten(),

            'token_type_ids':
                encoding['token_type_ids'].flatten()
        }

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        # ==================================================
        # ACL-ARC
        # ==================================================

        if self.dataset_type == 'acl-arc':

            text = item.get('text', '')

            intent_label = item.get('intent', '')

            section_label = item.get('section', '')

            worthiness_label = item.get(
                'worthiness',
                0
            )

        # ==================================================
        # SciCite
        # ==================================================

        else:

            text = item.get('string', '')

            raw_label = (
                item.get('label', '')
                .lower()
            )

            intent_label = (
                self.scicite_intent_map
                .get(raw_label, '')
            )

            section_label = self._normalize_section(
                item.get('sectionName', '')
            )

            worthiness_label = (
                1
                if item.get('isKeyCitation', False)
                else 0
            )

        # ==================================================
        # task-specific prompts
        # ==================================================

        intent_prompt = self.build_intent_prompt(
            text
        )

        section_prompt = self.build_section_prompt(
            text
        )

        worthiness_prompt = self.build_worthiness_prompt(
            text
        )

        # ==================================================
        # encoding
        # ==================================================

        intent_encoding = self.encode_text(
            intent_prompt
        )

        section_encoding = self.encode_text(
            section_prompt
        )

        worthiness_encoding = self.encode_text(
            worthiness_prompt
        )

        # ==================================================
        # labels
        # ==================================================

        intent_id = self.intent_label2id.get(
            intent_label,
            -1
        )

        section_id = self.section_label2id.get(
            section_label,
            -1
        )

        worthiness_id = int(worthiness_label)

        return {

            # ==================================================
            # intent
            # ==================================================

            'intent_input_ids':
                intent_encoding['input_ids'],

            'intent_attention_mask':
                intent_encoding['attention_mask'],

            'intent_token_type_ids':
                intent_encoding['token_type_ids'],

            # ==================================================
            # section
            # ==================================================

            'section_input_ids':
                section_encoding['input_ids'],

            'section_attention_mask':
                section_encoding['attention_mask'],

            'section_token_type_ids':
                section_encoding['token_type_ids'],

            # ==================================================
            # worthiness
            # ==================================================

            'worthiness_input_ids':
                worthiness_encoding['input_ids'],

            'worthiness_attention_mask':
                worthiness_encoding['attention_mask'],

            'worthiness_token_type_ids':
                worthiness_encoding['token_type_ids'],

            # ==================================================
            # labels
            # ==================================================

            'intent_label':
                torch.tensor(
                    intent_id,
                    dtype=torch.long
                ),

            'section_label':
                torch.tensor(
                    section_id,
                    dtype=torch.long
                ),

            'worthiness_label':
                torch.tensor(
                    worthiness_id,
                    dtype=torch.long
                )
        }


def create_dataloader(
    dataset,
    batch_size=40,
    shuffle=True
):

    return torch.utils.data.DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=4,

        pin_memory=True
    )