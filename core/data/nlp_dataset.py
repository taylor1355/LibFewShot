# Adapted from https://github.com/mailong25/meta-learning-bert

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset
from augmentations import eda, AugTransformer()

class NLPDataset(Dataset):

    def __init__(self, examples, labels, tokenizer, mode):
        self.examples = examples
        random.shuffle(self.examples)
        
        self.mode = mode
        
        if self.mode == "train":
            self.aug_transformer = AugTransformer()
            self.augment()

        self.example_labels = [ex['label'] for ex in examples]
        self.labels = list(set(self.example_labels))

        self.tokenizer = tokenizer
        self.max_seq_length = 256 # TODO: get this from the current model instead of hardcoding

    def augment(self):
        for i in range(len(self.examples)):
            ex = self.examples[i]
            new_ex = ex
            if random.randint(0, 5) <= 1:
                eda_exammples = eda(ex['raw'], alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.1, num_aug=3) 
                for aug in eda_exammples: # Don't include original which is at last position
                    new_ex['raw'] = aug
                    new_ex['text'] = aug.split()
                    self.examples.append(new_ex)
            if random.randint(0, 5) <= 1:
                translated = aug_transformer.backtranslate(ex['raw'])
                new_ex['raw'] = translated
                new_ex['text'] = translated.split()
                self.examples.append(new_ex)
            if random.randint(0, 5) == 0:
                str_arr = ex['raw'].split()
                snippet = " ".join(str_arr[:random.randint(0, len(str_arr)-1)])
                gen_text = aug_transformer.generate(snippet, 10)
                new_ex['raw'] = gen_text
                new_ex['text'] = gen_text.split()
                self.examples.append(new_ex)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.tokenizer.encode(example['raw'][:self.max_seq_length])
        attention_mask = [1] * len(input_ids)
        segment_ids    = [0] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)

        label_id = example['label']

        input_ids = torch.Tensor(input_ids).to(torch.long)
        attention_mask = torch.Tensor(attention_mask).to(torch.long)
        segment_ids = torch.Tensor(segment_ids).to(torch.long)
        label_id = torch.Tensor([label_id]).to(torch.long)

        return (input_ids, attention_mask, segment_ids), label_id

    def __len__(self):
        return len(self.examples)
