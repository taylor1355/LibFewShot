# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Sampler


class DomainsSampler(Sampler):
    """A Sampler to sample a FSL task.

    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
        self,
        domain_list,
        domain_num,
        episode_size,
        episode_num,
        way_num,
        data_num,
    ):
        """Init a DomainsSampler and generate a domain-index list.

        Args:
            domain_list (list): The domain list from domain list.
            domain_num (int): The number of unique domains.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            data_num (int): FSL setting.
        """
        super(DomainsSampler, self).__init__(domain_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.data_num = data_num

        domain_list = np.array(domain_list)
        self.idx_list = []
        for domain_idx in range(domain_num):
            ind = np.argwhere(domain_list == domain_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).

        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num):
            domains = torch.randperm(len(self.idx_list))[: self.way_num]
            for d in domains:
                idxes = self.idx_list[d.item()]
                pos = torch.randperm(idxes.size(0))[: self.data_num]
                batch.append(idxes[pos])
            print(f'Batchlen: {len(batch)}')
            print(f'Ep size: {self.episode_size}')
            print(f'Way num: {self.way_num}')
            if len(batch) == self.episode_size * self.way_num:
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []
