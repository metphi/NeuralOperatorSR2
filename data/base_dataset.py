# data/base_dataset.py
from abc import ABC, abstractmethod
import torch

class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, config, split):
        self.config = config
        self.split  = split          # 'train' / 'val' / 'test'
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """读取原始数据，初始化索引"""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """返回单条样本"""
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_collate_fn(self):
        """可选override：自定义batch组装逻辑"""
        return None