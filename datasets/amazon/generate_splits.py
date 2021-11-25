# Using splits from https://github.com/allenai/flex/blob/a31ef7ca672b6d7e5b543de6639e677cd5677ed7/fewshot/hf_datasets_scripts/amazon/amazon.py

import json
import os
from os.path import join

dataset_dir = os.path.dirname(__file__)
examples = []
with open(join(dataset_dir, 'amazon.json')) as data_file:
    examples = [json.loads(line) for line in data_file]
    for i, example in enumerate(examples):
        example['id'] = i

# overwrite original with added id field
with open(join(dataset_dir, 'amazon.json'), 'w') as data_file:
    json.dump(examples, data_file)

labels = [
    'Amazon_Instant_Video',
    'Apps_for_Android',
    'Automotive',
    'Baby',
    'Beauty',
    'Books',
    'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Grocery_and_Gourmet_Food',
    'Health_and_Personal_Care',
    'Home_and_Kitchen',
    'Kindle_Store',
    'Movies_and_TV',
    'Musical_Instruments',
    'Office_Products',
    'Patio_Lawn_and_Garden',
    'Pet_Supplies',
    'Sports_and_Outdoors',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Video_Games'
]
with open(join(dataset_dir, f'labels.txt'), 'w') as labels_file:
    json.dump(labels, labels_file)

splits = {
    'train': set([2, 3, 4, 7, 11, 12, 13, 18, 19, 20]),
    'val': set([1, 22, 23, 6, 9]),
    'test': set([0, 5, 14, 15, 8, 10, 16, 17, 21]),
}

for split_name, split_labels in splits.items():
    split_examples = [ex for ex in examples if ex['label'] in split_labels]
    with open(join(dataset_dir, f'amazon_{split_name}.json'), 'w') as split_file:
        json.dump(split_examples, split_file)
        print(f'Saved {len(split_examples)} {split_name} examples')
