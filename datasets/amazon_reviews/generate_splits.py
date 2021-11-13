import json
import os
from os.path import join

dataset_dir = os.path.dirname(__file__)
examples = json.load(open(join(dataset_dir, 'amazon_reviews.json')))
print(f'First example: {examples[0]}')

domains = set([example['domain'] for example in examples])
print(f'Domains: {domains}')

test_domains = ['cell_phones_&_service', 'office_products']
val_domains = ['dvd', 'computer_&_video_games', 'automotive']

example_splits = {}
example_splits['train'] = [ex for ex in examples if ex['domain'] not in test_domains and ex['domain'] not in val_domains]
example_splits['val'] = [ex for ex in examples if ex['domain'] in val_domains]
example_splits['test'] = [ex for ex in examples if ex['domain'] in test_domains]
print(f'{len(example_splits["train"])} training examples, {len(example_splits["val"])} validation examples, {len(example_splits["test"])} test examples')

for split_name, split in example_splits.items():
    with open(join(dataset_dir, f'amazon_reviews_{split_name}.json'), 'w') as split_file:
        json.dump(example_splits[split_name], split_file)
        print(f'Saved {split_name} examples')
