# data/loaders/reproducible_loader.py
import torch
from torch.utils.data import DataLoader
from data.base_dataloader import BaseLoader
from data.registry import LOADER_REGISTRY
import numpy as np
import random


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@LOADER_REGISTRY.register("reproducible")
class ReproducibleLoader(BaseLoader):
    """
    可复现 loader
    在 TorchDataLoader 基础上额外固定 generator 和 worker seed
    """

    def _build(self):
        cfg   = self.config.loader
        batch_size = getattr(cfg, f'{self.split}_batch_size', cfg.batch_size)
        shuffle    = (self.split == 'train') and cfg.get('shuffle_train', True)

        g = torch.Generator()
        g.manual_seed(self.config.data.seed)

        self._loader = DataLoader(
            self.dataset,
            batch_size     = batch_size,
            shuffle        = shuffle,
            num_workers    = cfg.num_workers,
            pin_memory     = cfg.pin_memory,
            worker_init_fn = _seed_worker,
            generator      = g,             
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)