# Adapted from https://github.com/mailong25/meta-learning-bert

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset

class NLPDataset(Dataset):

    def __init__(self, examples, labels, tokenizer):
        self.examples = examples
        random.shuffle(self.examples)

        self.example_labels = [ex['label'] for ex in examples]
        self.labels = list(set(self.example_labels))

        self.tokenizer = tokenizer
        self.max_seq_length = 256 # TODO: get this from the current model instead of hardcoding

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.tokenizer.encode(example['raw'][:self.max_seq_length], add_special_tokens=True)
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
