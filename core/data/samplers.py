# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Sampler


class CategoriesSampler(Sampler):
    """A Sampler to sample a FSL task.
    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
        self,
        example_labels,
        label_ids,
        episode_size,
        episode_num,
        way_num,
        data_num,
    ):
        """Init a CategoriesSampler and generate a label-index list.
        Args:
            example_labels (list): The label list from example list.
            label_ids (list): The list of unique label ids.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            data_num (int): FSL setting.
        """
        super(CategoriesSampler, self).__init__(example_labels)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.data_num = data_num

        self.use_new_example_list(example_labels, label_ids)

    def use_new_example_list(self, example_labels, label_ids):
        example_labels = np.array(example_labels)
        self.idx_list = []
        for label_idx in label_ids:
            ind = np.argwhere(example_labels == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)
        print(f'idx_list len: {list([len(i) for i in self.idx_list])}')

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).
        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list))[: self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[: self.data_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []
