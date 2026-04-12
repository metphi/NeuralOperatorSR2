# trainer/base_trainer.py
import math
import torch
import logging
from pathlib import Path
import time
import warnings
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):
    def __init__(self, model, loaders, config):
        self.model   = model
        self.loaders = loaders       # {'train': ..., 'val': ..., 'test': ...}
        self.config  = config
        self.start_epoch = 0
        self._setup()

    def _setup(self):
        """初始化optimizer, scheduler, scaler, logger, checkpoint"""
        self.save_path = Path(self.config.trainer.save_path) / f"{time.strftime('%Y%m%d_%H%M%S')}"
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._save_config()
        self._set_globel_seed()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.logger, self.log_path = self._build_logger()
        self.writer    = SummaryWriter(log_dir=str(self.save_path))
        self.best_eval_metric = float('inf')
        # 如果 config 里指定了 resume 路径，自动恢复
        self._set_device()
        if self.config.get("resume", None):
            self._resume(self.config.resume)

    # ---- 必须实现 --------------------------------------------------------
    @abstractmethod
    def _train_step(self, batch) -> dict:
        """
        单个 batch 的前向 + loss 计算。
        返回值中必须包含 _get_train_metric_keys() 所声明的所有 key，
        且参与 backward 的 loss 必须是 tensor（保留计算图）。
        例：return {'train_loss': loss_tensor, 'train_mse': mse_tensor}
        """
        pass

    @abstractmethod
    def _eval_step(self, batch) -> dict:
        """
        单个 batch 的评估计算（在 torch.no_grad() 内调用）。
        返回值中必须包含 _get_eval_metric_keys() 所声明的所有 key。
        例：return {'val_loss': loss_val, 'val_psnr': psnr_val}
        """
        pass

    @abstractmethod
    def _get_train_metric_keys(self) -> list:
        """
        声明训练阶段需要追踪的 metric 名称列表。
        第一个 key 默认作为 backward loss。
        例：return ['train_loss', 'train_mse', 'train_perceptual']
        """
        pass

    @abstractmethod
    def _get_eval_metric_keys(self) -> list:
        """
        声明验证阶段需要追踪的 metric 名称列表。
        例：return ['val_loss', 'val_psnr', 'val_ssim']
        """
        pass

    # ---- 默认实现（可 override）------------------------------------------

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info(f"Training started | epochs: {self.config.epochs}")
        self.logger.info("=" * 60)
        self.model.to(self.device)
        self.on_train_begin()
        for epoch in range(self.start_epoch, self.config.epochs):
            self.on_epoch_begin(epoch)
            self._train_epoch()
            self._valid_epoch()
            self.on_epoch_end(epoch)
        self.writer.close()
        self.on_train_end()
        self.logger.info("Training finished.")
        

    def _train_epoch(self):
        self._model_train_state_set()
        train_keys    = self._get_train_metric_keys()
        backward_key  = train_keys[0]           # 约定第一个 key 参与 backward
        total_samples = 0

        for batch in self.loaders['train']:
            batch_size = batch[0].shape[0]
            batch = [item.to(self.device) for item in batch]
            self.optimizer.zero_grad()

            out = self._train_step(batch)       # 子类实现，返回包含 tensor 的 dict

            # backward
            out[backward_key].backward()
            if self.config.get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()

            for k in train_keys:
                val = out[k].item() if isinstance(out[k], torch.Tensor) else out[k]
                self.train_metric[k] += val * batch_size
            total_samples += batch_size

        for k in train_keys:
            self.train_metric[k] /= total_samples

    def _valid_epoch(self):
        self._model_val_state_set()
        eval_keys     = self._get_eval_metric_keys()
        total_samples = 0

        with torch.no_grad():           
            for batch in self.loaders['val']:
                batch_size = batch[0].shape[0]
                batch = [item.to(self.device) for item in batch]
                out = self._eval_step(batch)    # 子类实现

                for k in eval_keys:
                    val = out[k].item() if isinstance(out[k], torch.Tensor) else out[k]
                    self.eval_metric[k] += val * batch_size
                total_samples += batch_size

        for k in eval_keys:
            self.eval_metric[k] /= total_samples

    # ---- 生命周期钩子（默认空实现，按需 override）------------------------

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass
    
    def _model_train_state_set(self):
        self.model.train()
        
    def _model_val_state_set(self):
        self.model.eval()

    def on_epoch_begin(self, epoch):
        self.current_epoch = epoch
        # 由子类声明的 key 动态初始化，不在 Base 里硬编码
        self.train_metric = {k: 0.0 for k in self._get_train_metric_keys()}
        self.eval_metric  = {k: 0.0 for k in self._get_eval_metric_keys()}

    def on_epoch_end(self, epoch):              # ← 签名与 train() 调用一致
        # scheduler 按 epoch 更新（语义清晰；若需 warmup by step，在子类 override）
        self.scheduler.step()
        # log 到控制台 / 文件
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.config.epochs}]" +
            f" | lr: {self.current_lr:.2e}" +
            "".join(f" | {k}: {v:.4f}" for k, v in self.train_metric.items()) +
            "".join(f" | {k}: {v:.4f}" for k, v in self.eval_metric.items())
        )
        # log 到 TensorBoard
        for k, v in self.train_metric.items():
            self.writer.add_scalar(f"train/{k}", v, epoch + 1)
        for k, v in self.eval_metric.items():
            self.writer.add_scalar(f"val/{k}", v, epoch + 1)

        # checkpoint
        self._save_checkpoint()

    # ---- 通用工具（所有子类共享）----------------------------------------

    def _build_optimizer(self):
        opt_cfg = self.config.optimizer                     # 统一用 self.config
        optimizer_type = opt_cfg.get("type", "adam").lower()

        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr           = self.config.lr,
                betas        = tuple(opt_cfg.get("betas", (0.9, 0.999))),
                eps          = opt_cfg.get("eps", 1e-8),
                weight_decay = opt_cfg.get("weight_decay", 0.0),
                amsgrad      = opt_cfg.get("amsgrad", False),
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr           = self.config.lr,
                betas        = tuple(opt_cfg.get("betas", (0.9, 0.999))),
                eps          = opt_cfg.get("eps", 1e-8),
                weight_decay = opt_cfg.get("weight_decay", 1e-2),
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr           = self.config.lr,
                momentum     = opt_cfg.get("momentum", 0.9),
                weight_decay = opt_cfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_type}'. "
                f"Choose from ['adam', 'adamw', 'sgd']."
            )

    def _build_scheduler(self):
        cfg           = self.config
        lr_peak       = cfg.lr
        warmup_epochs = cfg.scheduler.warmup_epochs
        lr_min        = cfg.scheduler.lr_min

        # LambdaLR 的 epoch 参数实际上是 scheduler.step() 的调用次数
        # 这里按 epoch 语义设计，与 on_epoch_end 中的 scheduler.step() 对应
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup
                return (lr_min / lr_peak) + (1.0 - lr_min / lr_peak) * epoch / max(warmup_epochs, 1)
            # cosine annealing
            progress  = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
            cos_val   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (lr_min / lr_peak) + (1.0 - lr_min / lr_peak) * cos_val

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _build_logger(self):
        log_path = self.save_path / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger   = logging.getLogger(f"TrainLogger_{id(self)}")  # 避免多实例共享 handler
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger, log_path

    def _set_device(self):
        self.device = self.config.get("device", 'cpu')
        if not torch.backends.cudnn.is_available() and str(self.device) != 'cpu':
            warnings.warn(
                f"cuDNN is not acceptable/available on this system. "
                f"Switching device from {self.device} to cpu.",
                UserWarning
            )
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device)


    def _save_checkpoint(self):
        eval_metric_key = self.config.get('eval_metric_key', 'val_loss')
        checkpoint = {
            "epoch"               : self.current_epoch + 1,
            "lr"                  : self.current_lr,
            "model_state_dict"    : self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_metric"    : self.best_eval_metric,        
            "train_metric"        : self.train_metric,
            "eval_metric"         : self.eval_metric,
        }
        # 始终保存 last
        torch.save(checkpoint, self.save_path / "last.pth")
        # 按主指标保存 best
        current = self.eval_metric.get(eval_metric_key, float('inf'))
        if current <= self.best_eval_metric:
            self.best_eval_metric = current
            torch.save(checkpoint, self.save_path / "best.pth")
            self.logger.info(
                f"  ↑ Best checkpoint saved | {eval_metric_key}: {current:.5f}"
            )

    def _save_config(self):
        OmegaConf.save(self.config, self.save_path / "config.yaml")
        self._ALL_CONFIG = self.config  # 方便子类访问全部 config，避免过度拆分子 config 导致访问不便
        self.config = self.config.trainer

    def _set_globel_seed(self):
        import numpy as np
        import random
        seed = self.config.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    
    def _resume(self, path):
        self.logger.info(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch      = checkpoint['epoch']        # 从下一个 epoch 继续
        self.best_eval_metric = checkpoint.get('best_eval_metric', float('inf'))
        self.logger.info(
            f"Resumed | start_epoch: {self.start_epoch} "
            f"| best_eval_metric: {self.best_eval_metric:.5f}"
        )