# data/base_dataloader.py
# data/base_dataloader.py
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    所有 Loader 的抽象基类
    - 本身就是一个 iterable，可以直接 for batch in loader
    - 强制子类暴露 dataset、set_epoch 等接口
    """

    def __init__(self, dataset, config, ):
        self.dataset = dataset
        self.split = self.dataset.split
        self.config  = config
        self._build()           # 子类在这里完成内部 loader 的构造

    @abstractmethod
    def _build(self):
        """构造内部迭代逻辑，子类必须实现"""
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def set_epoch(self, epoch: int):
        """默认实现：通知 dataset；子类可 override 追加额外逻辑"""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)