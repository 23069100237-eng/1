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
                if len(token_ids) > 0:
                    ids.append(token_ids[0])
            ids = list(set(ids))
            self.label_word_ids[label] = ids

    def project(self, mask_logits):

        probs = F.softmax(mask_logits, dim=-1)

        class_scores = []

        for label, word_ids in self.label_word_ids.items():

            valid_ids = []

            for wid in word_ids:

                if wid != self.tokenizer.unk_token_id:
                    valid_ids.append(wid)

            if len(valid_ids) == 0:

                score = torch.zeros(
                    probs.size(0),
                 device=probs.device
            )

            else:

                score = probs[:, valid_ids].mean(dim=1)

            class_scores.append(score)

        class_scores = torch.stack(
            class_scores,
            dim=1
    )

        return torch.log(class_scores + 1e-12)