# Adapted from https://github.com/mailong25/meta-learning-bert

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset

LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}

class NLPDataset(Dataset):

    # TODO: remove unused parameters
    def __init__(self, examples, num_task, k_support, k_query, tokenizer):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        random.shuffle(self.examples)

        # TODO: add a domain_index field to the dataset itself, so that this logic still works when there are actually train/val/test splits
        self.unique_domains = list(set([example['domain'] for example in examples]))
        self.domain_num = len(self.unique_domains)
        self.domain_indices = {domain: index for (index, domain) in enumerate(self.unique_domains)}
        self.example_domain_indices = [self.domain_indices[example['domain']] for example in examples]

        self.tokenizer = tokenizer
        self.max_seq_length = 256 # TODO: get this from the current model instead of hardcoding

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.tokenizer.encode(example['text'])
        attention_mask = [1] * len(input_ids)
        segment_ids    = [0] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)

        label_id = LABEL_MAP[example['label']]

        input_ids = torch.Tensor(input_ids).to(torch.long)
        attention_mask = torch.Tensor(attention_mask).to(torch.long)
        segment_ids = torch.Tensor(segment_ids).to(torch.long)
        label_id = torch.Tensor([label_id]).to(torch.long)

        return (input_ids, attention_mask, segment_ids), label_id

    def __len__(self):
        return len(self.examples)
