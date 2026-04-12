# data/datasets/div2k_dataset.py

from pathlib import Path
from PIL import Image
import random
import torchvision.transforms.functional as TF
from data.base_dataset import BaseDataset
from data.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DIV2K")
class DIV2KDataset(BaseDataset):
    

    """config.data中需要有：
    scale
    patch_h
    patch_w
    seed
    data_path
    """
    
    def _load_data(self):
        # ---- 从 config 读取所有参数 ----
        cfg          = self.config.data          # 数据相关子config
        norm = cfg.get("norm", None)                  
        if norm is None:
            self.norm = {'lr': {'sub': 0, 'div': 1}, 'gt': {'sub': 0, 'div': 1}}
        else:
            self.norm = norm
        train = True if self.split=='train' else False
        self.s       = cfg.scale
        self.patch_h = cfg.patch_h
        self.patch_w = cfg.patch_w
        self.seed    = cfg.seed

        if train:
            self.lr_root = Path(cfg.data_path) /"DIV2K_train_LR_bicubic"/f'X{self.s}'
            self.hr_root = Path(cfg.data_path) /"DIV2K_train_HR"
        else:
            self.lr_root = Path(cfg.data_path) /"DIV2K_valid_LR_bicubic"/f'X{self.s}'
            self.hr_root = Path(cfg.data_path) /"DIV2K_valid_HR"
   
        self.start   = 1 if train else 801
        self.count   = 800 if train else 100

        self.epoch   = 0                         # 由 Trainer 通过 set_epoch() 注入

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # ---- 读图 ----
        lr_path = self.lr_root / f'{idx + self.start:04d}x{self.s}.png'
        hr_path = self.hr_root / f'{idx + self.start:04d}.png'

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        # ---- 确定性随机裁剪（seed + epoch + idx 三元组保证可复现）----
        if self.split == 'train':
            rng   = random.Random(self.seed * 10_000_019 + self.epoch * 1_000_003 + idx)
            lr_w, lr_h = lr_image.size          # PIL: (W, H)
            max_h = lr_h - self.patch_h
            max_w = lr_w - self.patch_w
            top   = rng.randint(0, max_h)
            left  = rng.randint(0, max_w)
            # ---- 裁剪（LR和HR对齐）----
            lr_crop = TF.crop(lr_image, top,          left,          self.patch_h,          self.patch_w)
            hr_crop = TF.crop(hr_image, top  * self.s, left * self.s, self.patch_h * self.s, self.patch_w * self.s)
            normed_lr_crop = (TF.to_tensor(lr_crop) - self.norm['lr']['sub']) / self.norm['lr']['div']
            normed_hr_crop = (TF.to_tensor(hr_crop) - self.norm['gt']['sub']) / self.norm['gt']['div']
            return normed_lr_crop, normed_hr_crop
        else:
            # 验证阶段不裁剪，直接返回全图
            normed_lr_image = (TF.to_tensor(lr_image) - self.norm['lr']['sub']) / self.norm['lr']['div']
            normed_hr_image = (TF.to_tensor(hr_image) - self.norm['gt']['sub']) / self.norm['gt']['div']
            return normed_lr_image, normed_hr_image