# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from transformers import AutoTokenizer

from core.data.dataset import GeneralDataset
from core.data.nlp_dataset import NLPDataset
from .collates import get_collate_function, get_augment_method
from .samplers import CategoriesSampler
from ..utils import ModelType

from pathlib import Path
from os.path import join
import os
import json

def get_dataloader(config, mode, model_type):
    """Get the dataloader corresponding to the model type and training phase.

    According to the config dict, the training phase and model category, select the appropriate transforms, set the corresponding sampler and collate_fn, and return the corresponding dataloader.

    Args:
        config (dict): A LibFewShot setting dict
        mode (str): mode in train/test/val
        model_type (ModelType): model type in meta/metric//finetuning

    Returns:
        Dataloader: The corresponding dataloader.
    """
    assert model_type != ModelType.ABSTRACT

    # Does not currently work with finetuning models (ie model_type == ModelType.FINETUNING). And it has only been tested with proto_net
    # These methods expect the dataloader to return single samples, instead the dataloader returns
    # few shot tasks (ie batches of samples)

    project_root = Path(__file__).parents[2] # 2 directories up from this file's directory
    dataset_dir = join(project_root, config['data_root'])

    examples = json.load(open(join(dataset_dir, f'amazon_{mode}.json')))
    labels = json.load(open(join(dataset_dir, 'labels.txt')))

    augmentations = []
    one_to_one_augmentation=False
    if config['augmentation']['enabled']:
        augmentation_path = f'{join(dataset_dir, config["augmentation"]["file_prefix"])}_{mode}.json'
        if config['augmentation']['type'] == 'append' and mode == 'train':
            print('Using append augmentation mode. Augmentations will be appended according to temperature to training data.')
            augmentations = json.load(open(augmentation_path))
        elif config['augmentation']['type'] == 'replace' and mode == 'train':
            print('Using replace augmentation mode. Training data will be replaced with augmentations.')
            examples = json.load(open(augmentation_path))
        elif config['augmentation']['type'] == 'support':
            print('Using support augmentation mode. Examples in support sets of training, validation, and test data will be augmented with corresponding augmented versions.')
            augmentations = json.load(open(augmentation_path))
            one_to_one_augmentation = True

    tokenizer = AutoTokenizer.from_pretrained(config['backbone']['kwargs']['bert_model'], do_lower_case = True) # TODO: take in tokenizer instead of constructing here
    dataset = NLPDataset(examples, labels, tokenizer, augmentations=augmentations, one_to_one_augmentation=one_to_one_augmentation)

    sampler = CategoriesSampler(
        example_labels=dataset.get_combined_example_labels(),
        label_ids=dataset.labels,
        episode_size=config["episode_size"],
        episode_num=config["train_episode"] if mode == "train" else config["test_episode"],
        way_num=config["way_num"] if mode == "train" else config["test_way"],
        data_num=(config["shot_num"] + config["query_num"]) if mode == "train" else (config["test_shot"] + config["test_query"])
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,#config["n_gpu"] * 4, # Needs to be set to 0 to work on Windows
        pin_memory=True,
        collate_fn=lambda batches: tuple(zip(*batches)),
    )

    return dataloader

def update_dataloader_temperature(dataloader, config, new_temp):
    dataloader.dataset.update_temperature(new_temp)

    example_labels = dataloader.dataset.get_combined_example_labels()
    label_ids = dataloader.dataset.labels
    dataloader.batch_sampler.use_new_example_list(example_labels, label_ids)
