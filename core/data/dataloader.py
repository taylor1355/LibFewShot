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

    # TODO: move hardcoded stuff like paths into configs

    dataset_dir = join(project_root, 'datasets/amazon')
    if mode == "train":
        examples = json.load(open(join(dataset_dir, f'no_id/amazon_eda_train.json')))
    else:
        examples = json.load(open(join(dataset_dir, f'amazon_{mode}.json')))
    labels = json.load(open(join(dataset_dir, 'labels.txt')))

    tokenizer = AutoTokenizer.from_pretrained(config['backbone']['kwargs']['bert_model'], do_lower_case = True) # TODO: take in tokenizer instead of constructing here
    dataset = NLPDataset(examples, labels, tokenizer)

    sampler = CategoriesSampler(
        example_labels=dataset.example_labels,
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
