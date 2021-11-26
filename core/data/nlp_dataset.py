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

    def __init__(self, examples, labels, tokenizer, augmentations=None):
        self.examples = examples
        self.augmentations = [] if augmentations is None else augmentations
        random.shuffle(self.examples)
        random.shuffle(self.augmentations)

        self.temp = 0
        self.num_used_augmentations = 0

        self.aug_transformer = None

        self.example_labels = [ex['label'] for ex in examples]
        self.labels = list(set(self.example_labels))

        self.tokenizer = tokenizer
        self.max_seq_length = 256

    def init_aug_transformer_if_needed():
        if self.aug_transformer is None:
            self.aug_transformer = AugTransformer()

    def augment(self):
        self.examples.extend(self.generate_augmentations(self.generate_augmentation_random))

    def generate_eda_augmentation(self, text):
        return eda(text, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.1, num_aug=3)

    def generate_backtranslate_augmentation(self, text):
        self.init_aug_transformer_if_needed()
        return [self.aug_transformer.backtranslate(text)]

    def generate_gen_transformer_augmentation(self, text):
        self.init_aug_transformer_if_needed()
        str_arr = text.split()
        snippet = " ".join(str_arr[:random.randint(0, len(str_arr)-1)])
        return [self.aug_transformer.generate(snippet, 10)]

    def generate_augmentation_random(self, text):
        augments = []
        if random.randint(0, 5) <= 1:
            augments.extend(self.generate_eda_augmentation(text))
        if random.randint(0, 5) <= 1:
            augments.extend(self.generate_backtranslate_augmentation(text))
        if random.randint(0, 5) == 0:
            augments.extend(self.generate_gen_transformer_augmentation(text))
        return augments

    def generate_augmentations(self, augmentation_method):
        augments = []
        for i, ex in enumerate(self.examples):
            print(f'Generating augmented example {i}/{len(self.examples)}')
            ex_augments = augmentation_method(ex['raw'])
            for aug in ex_augments:
                new_ex = ex.copy()
                new_ex['raw'] = aug
                new_ex['text'] = aug.split()
                augments.append(new_ex)
        return augments

    def update_temperature(self, new_temp):
        self.temp = new_temp
        self.num_used_augmentations = int(self.temp * len(self.examples))
        print(f'Temperature set to {new_temp}. Using {self.num_used_augmentations}/{len(self.augmentations)} augmentations.')

    def __getitem__(self, index):
        if index < len(self.examples):
            example = self.examples[index]
        else:
            index -= len(self.examples)
            example = self.augmentations[index % len(self.augmentations)]

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
        return len(self.examples) + self.num_used_augmentations
