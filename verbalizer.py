import torch
import torch.nn.functional as F

#verbalizer.py
class Verbalizer:

    def __init__(self, tokenizer, label_words):

        self.tokenizer = tokenizer
        self.label_words = label_words

        self.label_word_ids = {}

        for label, words in label_words.items():

            ids = []

            for word in words:

                token_ids = tokenizer.encode(
                    word,
                    add_special_tokens=False
                )

                # 只保留单token
                if len(token_ids) == 1:
                    ids.append(token_ids[0])

            self.label_word_ids[label] = ids

    def project(self, mask_logits):

        class_logits = []

        for label, word_ids in self.label_word_ids.items():

            if len(word_ids) == 0:

                score = torch.zeros(
                    mask_logits.size(0),
                    device=mask_logits.device
                )

            else:

            # 不做 softmax
                score = mask_logits[:, word_ids].mean(dim=1)

            class_logits.append(score)

        class_logits = torch.stack(
            class_logits,
            dim=1
    )

        return class_logits