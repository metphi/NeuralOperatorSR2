import math
import torch
from utils._utils import *
from loss.loss import FreqLoss, ResExplicitLoss
from trainer.base_trainer import BaseTrainer
from trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("CIStrainer")
class CIStrainer(BaseTrainer):

    def __init__(self, model, loaders, config):
        self.CISList = ["CISFNO", "CISModel", "CISFNO1", "CISFNO2", "CISFNO3"]     
        super().__init__(model, loaders, config)                          

    def _setup(self):
        super()._setup()
        self.alpha     = self.config.get("alpha", 1.0)
        self.psnr      = PSNR()
        self.ssim      = SSIM()
        self.ssim.to(self.device)
        self.criterion_head = nn.MSELoss()
        self.criterion_res = nn.L1Loss()

    def on_train_begin(self):
        super().on_train_begin()
        if self.model.__class__.__name__ not in self.CISList:
            raise TypeError(f"传入模型应该为{self.CISList}，got {self.model.__class__.__name__}")

    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        for split in self.loaders:
            sampler = self.loaders[split].dataset
            if hasattr(sampler, 'set_epoch'):        
                sampler.set_epoch(self.current_epoch)

    def _train_step(self, batch):
        lr_batch, hr_batch = batch
        head_pred, res_pred = self.model(lr_batch)
        gt_res = hr_batch - head_pred
        loss1  = self.criterion_head(head_pred, hr_batch)
        loss2  = self.criterion_res(res_pred, gt_res.detach())
        loss   = loss1 + self.alpha * loss2
        return {
            'train_loss': loss,
            "head_loss" : loss1.detach(),
            "res_loss"  : loss2.detach(),
        }

    def _eval_step(self, batch):
        lr_batch, hr_batch = batch
        sr_batch = self.patch_inference(self.model, lr_batch, scale=self.model.scale)
        return {
            "val_loss": self.criterion_head(sr_batch, hr_batch).item(),
            "psnr"    : self.psnr(sr_batch, hr_batch).item(),
            "ssim"    : self.ssim(sr_batch, hr_batch).item(),
        }

    def _get_train_metric_keys(self):
        return ["train_loss", "head_loss", "res_loss"]

    def _get_eval_metric_keys(self):
        return ["val_loss", "psnr", "ssim"]
    
    def _save_checkpoint(self):
        eval_metric_key = self.config.get('eval_metric_key', 'val_loss')
        if eval_metric_key == "psnr" or eval_metric_key == "ssim":
            if self.current_epoch == self.start_epoch:
                self.best_eval_metric = -float('inf')
            checkpoint = {
                "epoch"               : self.current_epoch + 1,
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
            if current >= self.best_eval_metric:
                self.best_eval_metric = current
                torch.save(checkpoint, self.save_path / "best.pth")
                self.logger.info(
                    f"  ↑ Best checkpoint saved | {eval_metric_key}: {current:.5f}"
                )
        else:
                super()._save_checkpoint()
        
    def patch_inference(self, model, lr_tensor, patch_size=128, overlap=16, scale=2):
        """
        lr_tensor: (1, 3, H, W), range [0,1]
        """
        B, C, H, W = lr_tensor.shape
        stride = patch_size - overlap
        
        sr_tensor = torch.zeros(B, C, H*scale, W*scale, device=lr_tensor.device)
        count_map = torch.zeros_like(sr_tensor)

        for top in range(0, H, stride):
            for left in range(0, W, stride):
                t = min(top,  H - patch_size)
                l = min(left, W - patch_size)

                patch = lr_tensor[:, :, t:t+patch_size, l:l+patch_size]

                with torch.no_grad():
                    head_pred, res_pred = model(patch)  # ← 解包两个输出
                    sr_patch = head_pred + res_pred      # ← 真实预测

                sr_tensor[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += sr_patch
                count_map[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += 1

        return sr_tensor / count_map.clamp(min=1)
    
    
    
@TRAINER_REGISTRY.register("CIStrainer2")
class CIStrainer2(BaseTrainer):

    def __init__(self, model, loaders, config):
        self.CISList = ["CISFNO", "CISModel", "CISFNO1", "CISFNO2", "CISFNO3", "CISFNO4"]     
        super().__init__(model, loaders, config)                          

    def _setup(self):
        super()._setup()
        self.alpha     = self.config.get("alpha", 1.0)
        self.t0 = self.config.loss.get("t0", 250)
        self.psnr      = PSNR()
        self.ssim      = SSIM()
        self.ssim.to(self.device)
        self.criterion_head = nn.MSELoss()
        self.criterion_fuse = nn.MSELoss()
        self.criterion_pixel_res = nn.MSELoss()
        self.criterion_freq_res  = FreqLoss(
            high_freq_weight=self.config.loss.get("high_freq_weight", 2.0),
            low_freq_weight=self.config.loss.get("low_freq_weight", 1.0),
            low_freq_ratio=self.config.loss.get("low_freq_ratio", 0.3)
        )

    def _get_loss_weights(self, epoch, t0):
        # w1: 线性从1降到0，t0时归零
        w1 = 1.3 * max(0.0, 1.0 - epoch / t0)
        
        # w2: 在[0.9*t0, t0]线性从0升到1
        w2_start = 0.9 * t0
        if epoch <= w2_start:
            w2 = 0.0
        elif epoch >= t0:
            w2 = 1.0
        else:
            w2 = (epoch - w2_start) / (t0 - w2_start)
            
        w3 = 1.0 #res_pixel和res_freq的权重设置
        
        return w1, w2, w3
    
    def _build_scheduler(self):
        """
        分两段：
        - [0, t0]:       warmup(5epoch) + cosine 下降到 min_lr
        - [t0, total]:   restart 到 base_lr * restart_lr_ratio，再 cosine 下降到 min_lr
        restart_lr_ratio: t0处重启的lr相对base_lr的比例，默认0.3
        """
        total_epochs = self.config.epochs
        t0 = self.config.loss.get("t0", 250)
        base_lr = self.config.lr
        min_lr = self.config.scheduler.get("min_lr", 1e-5)
        restart_lr_ratio = self.config.scheduler.get("restart_lr_ratio", 0.3)

        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                # 线性warmup
                return epoch / warmup_epochs
            
            if epoch < t0:
                # 第一段cosine
                progress = (epoch - warmup_epochs) / (t0 - warmup_epochs)
                cosine   = 0.5 * (1 + math.cos(math.pi * progress))
                # 从1降到min_lr/base_lr
                lo = min_lr / base_lr
                return lo + (1 - lo) * cosine
            
            else:
                # 第二段：t0处重启，短warmup(3epoch)后cosine下降
                restart_warmup = self.config.scheduler.get("restart_warmup", 3)
                elapsed  = epoch - t0
                peak     = restart_lr_ratio          # 重启峰值比例
                lo       = min_lr / base_lr

                if elapsed < restart_warmup:
                    # 从min_lr线性warmup到peak
                    return lo + (peak - lo) * (elapsed / restart_warmup)
                else:
                    progress = (elapsed - restart_warmup) / (total_epochs - t0 - restart_warmup)
                    progress = min(progress, 1.0)
                    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
                    return lo + (peak - lo) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def on_train_begin(self):
        super().on_train_begin()
        if self.model.__class__.__name__ not in self.CISList:
            raise TypeError(f"传入模型应该为{self.CISList}，got {self.model.__class__.__name__}")

    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        for split in self.loaders:
            sampler = self.loaders[split].dataset
            if hasattr(sampler, 'set_epoch'):        
                sampler.set_epoch(self.current_epoch)

    def _train_step(self, batch):
        lr_batch, hr_batch = batch
        head_pred, res_pred = self.model(lr_batch)
        head_res = hr_batch - head_pred
        w1, w2, w3 = self._get_loss_weights(self.current_epoch, self.t0)
        loss_head = self.criterion_head(head_pred, hr_batch)
        loss_fuse = self.criterion_fuse(head_pred + res_pred, hr_batch)
        loss_res_pixel = self.criterion_pixel_res(res_pred, head_res.detach())
        loss_res_freq  = self.criterion_freq_res(res_pred, head_res.detach())
        loss_res = loss_res_pixel + w3 * loss_res_freq
        loss = loss_fuse + w1 * loss_head + w2 * loss_res
        
        return {
            'train_loss': loss,
            "head_loss" : loss_head.detach(),
            "fuse_loss" : loss_fuse.detach(),
            "res_loss_pixel": loss_res_pixel.detach(),
            "res_loss_freq" : loss_res_freq.detach(),
            "w1" : w1,
            "w2" : w2,
            "w3" : w3,
        }

    def _eval_step(self, batch):
        lr_batch, hr_batch = batch
        sr_batch = self.patch_inference(self.model, lr_batch, scale=self.model.scale)
        return {
            "val_loss": self.criterion_head(sr_batch, hr_batch).item(),
            "psnr"    : self.psnr(sr_batch, hr_batch).item(),
            "ssim"    : self.ssim(sr_batch, hr_batch).item(),
        }

    def _get_train_metric_keys(self):
        return ["train_loss", "head_loss", "fuse_loss", "res_loss_pixel", "res_loss_freq", "w1", "w2", "w3"]

    def _get_eval_metric_keys(self):
        return ["val_loss", "psnr", "ssim"]
    
    def _save_checkpoint(self):
        eval_metric_key = self.config.get('eval_metric_key', 'val_loss')
        if eval_metric_key == "psnr" or eval_metric_key == "ssim":
            if self.current_epoch == self.start_epoch:
                self.best_eval_metric = -float('inf')
            checkpoint = {
                "epoch"               : self.current_epoch + 1,
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
            if current >= self.best_eval_metric:
                self.best_eval_metric = current
                torch.save(checkpoint, self.save_path / "best.pth")
                self.logger.info(
                    f"  ↑ Best checkpoint saved | {eval_metric_key}: {current:.5f}"
                )
        else:
                super()._save_checkpoint()
        
    def patch_inference(self, model, lr_tensor, patch_size=128, overlap=16, scale=2):
        """
        lr_tensor: (1, 3, H, W), range [0,1]
        """
        B, C, H, W = lr_tensor.shape
        stride = patch_size - overlap
        
        sr_tensor = torch.zeros(B, C, H*scale, W*scale, device=lr_tensor.device)
        count_map = torch.zeros_like(sr_tensor)

        for top in range(0, H, stride):
            for left in range(0, W, stride):
                t = min(top,  H - patch_size)
                l = min(left, W - patch_size)

                patch = lr_tensor[:, :, t:t+patch_size, l:l+patch_size]

                with torch.no_grad():
                    head_pred, res_pred = model(patch)  # ← 解包两个输出
                    sr_patch = head_pred + res_pred      # ← 真实预测

                sr_tensor[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += sr_patch
                count_map[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += 1

        return sr_tensor / count_map.clamp(min=1)
    
    
    
    
@TRAINER_REGISTRY.register("CIStrainer3")
class CIStrainer3(BaseTrainer):

    def __init__(self, model, loaders, config):
        self.CISList = ["CISFNO", "CISModel", "CISFNO1", "CISFNO2", "CISFNO3", "CISFNO4"]     
        super().__init__(model, loaders, config)                          

    def _setup(self):
        super()._setup()
        self.psnr      = PSNR()
        self.ssim      = SSIM()
        self.ssim.to(self.device)
        self.epochs = self.config.epochs
        w1_end_ratio = self.config.loss.get("w1_end_ratio", 0.6)
        self.w1_end = self.epochs * w1_end_ratio
        w2_start_ratio = self.config.loss.get("w2_start_ratio", 0.2)
        w2_end_ratio = self.config.loss.get("w2_end_ratio", 0.8)
        self.w_start = self.epochs * w2_start_ratio
        self.w_end = self.epochs * w2_end_ratio
        self.criterion_head = nn.MSELoss()
        self.criterion_fuse = nn.MSELoss()
        self.criterion_res  = ResExplicitLoss(
            high_freq_weight=self.config.loss.get("high_freq_weight", 2.0),
            low_freq_weight=self.config.loss.get("low_freq_weight", 1.0),
            low_freq_ratio=self.config.loss.get("low_freq_ratio", 0.3)
        )

    def _get_loss_weights(self,):
        # w1: 从1缓慢线性下降到0，在total_epochs的大约一半归零
        w1 = max(0.0, 1.0 - self.current_epoch / self.w1_end)

        # w(t): 从极小值缓慢增大到1，sigmoid形状比线性更平滑
        if self.current_epoch < self.w_start:
            w = 0.01                         # 极小初值，不为0避免完全没有信号
        else:
            progress = (self.current_epoch - self.w_start) / (self.w_end - self.w_start)
            progress = min(progress, 1.0)
            # sigmoid 形状，过渡比线性更平滑
            w = 0.01 + 0.99 / (1 + math.exp(-10 * (progress - 0.5)))

        return w1, w
    
    
    def on_train_begin(self):
        super().on_train_begin()
        if self.model.__class__.__name__ not in self.CISList:
            raise TypeError(f"传入模型应该为{self.CISList}，got {self.model.__class__.__name__}")

    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        for split in self.loaders:
            sampler = self.loaders[split].dataset
            if hasattr(sampler, 'set_epoch'):        
                sampler.set_epoch(self.current_epoch)

    def _train_step(self, batch):
        lr_batch, hr_batch = batch
        head_pred, res_pred = self.model(lr_batch)
        head_res = (hr_batch - head_pred).detach()
        w1, w2 = self._get_loss_weights()
        loss_head = self.criterion_head(head_pred, hr_batch)
        loss_fuse = self.criterion_fuse(head_pred + res_pred, hr_batch)
        loss_res, loss_res_pixel, loss_res_freq = self.criterion_res(res_pred, head_pred, hr_batch)
        loss = loss_fuse + w1 * loss_head + w2 * loss_res
        
        return {
            'train_loss': loss,
            "head_loss" : loss_head.detach(),
            "fuse_loss" : loss_fuse.detach(),
            "res_loss_pixel": loss_res_pixel.detach(),
            "res_loss_freq" : loss_res_freq.detach(),
            "w1" : w1,
            "w2" : w2,
        }

    def _eval_step(self, batch):
        lr_batch, hr_batch = batch
        sr_batch = self.patch_inference(self.model, lr_batch, scale=self.model.scale)
        return {
            "val_loss": self.criterion_head(sr_batch, hr_batch).item(),
            "psnr"    : self.psnr(sr_batch, hr_batch).item(),
            "ssim"    : self.ssim(sr_batch, hr_batch).item(),
        }

    def _get_train_metric_keys(self):
        return ["train_loss", "head_loss", "fuse_loss", "res_loss_pixel", "res_loss_freq", "w1", "w2"]

    def _get_eval_metric_keys(self):
        return ["val_loss", "psnr", "ssim"]
    
    def _save_checkpoint(self):
        eval_metric_key = self.config.get('eval_metric_key', 'val_loss')
        if eval_metric_key == "psnr" or eval_metric_key == "ssim":
            if self.current_epoch == self.start_epoch:
                self.best_eval_metric = -float('inf')
            checkpoint = {
                "epoch"               : self.current_epoch + 1,
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
            if current >= self.best_eval_metric:
                self.best_eval_metric = current
                torch.save(checkpoint, self.save_path / "best.pth")
                self.logger.info(
                    f"  ↑ Best checkpoint saved | {eval_metric_key}: {current:.5f}"
                )
        else:
                super()._save_checkpoint()
        
    def patch_inference(self, model, lr_tensor, patch_size=128, overlap=16, scale=2):
        """
        lr_tensor: (1, 3, H, W), range [0,1]
        """
        B, C, H, W = lr_tensor.shape
        stride = patch_size - overlap
        
        sr_tensor = torch.zeros(B, C, H*scale, W*scale, device=lr_tensor.device)
        count_map = torch.zeros_like(sr_tensor)

        for top in range(0, H, stride):
            for left in range(0, W, stride):
                t = min(top,  H - patch_size)
                l = min(left, W - patch_size)

                patch = lr_tensor[:, :, t:t+patch_size, l:l+patch_size]

                with torch.no_grad():
                    head_pred, res_pred = model(patch)  # ← 解包两个输出
                    sr_patch = head_pred + res_pred      # ← 真实预测

                sr_tensor[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += sr_patch
                count_map[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += 1

        return sr_tensor / count_map.clamp(min=1)
    
    
    
@TRAINER_REGISTRY.register("CIStrainer4")
class CIStrainer4(BaseTrainer):

    def __init__(self, model, loaders, config):
        self.CISList = ["CISFNO", "CISModel", "CISFNO1", "CISFNO2", "CISFNO3", "CISFNO4"]     
        super().__init__(model, loaders, config)                          

    def _setup(self):
        super()._setup()
        self.alpha     = self.config.get("alpha", 1.0)
        self.t0 = self.config.loss.get("t0", 250)
        self.psnr      = PSNR()
        self.ssim      = SSIM()
        self.ssim.to(self.device)
        self.criterion_head = nn.MSELoss()
        self.criterion_fuse = nn.MSELoss()
        self.criterion_pixel_res = nn.MSELoss()
        self.criterion_res  = ResExplicitLoss(
            high_freq_weight=self.config.loss.get("high_freq_weight", 2.0),
            low_freq_weight=self.config.loss.get("low_freq_weight", 1.0),
            low_freq_ratio=self.config.loss.get("low_freq_ratio", 0.3)
        )
        
        self.epochs = self.config.epochs
        w1_end_ratio = self.config.loss.get("w1_end_ratio", 0.6)
        self.w1_end = self.epochs * w1_end_ratio
        self.t0 = self.w1_end
        w2_start_ratio = self.config.loss.get("w2_start_ratio", 0.2)
        w2_end_ratio = self.config.loss.get("w2_end_ratio", 0.8)
        self.w_start = self.epochs * w2_start_ratio
        self.w_end = self.epochs * w2_end_ratio
        
        

    def _get_loss_weights(self,):
        # w1: 从1缓慢线性下降到0，在total_epochs的大约一半归零
        w1 = 1.0 - self.current_epoch / self.w1_end if self.current_epoch < self.w1_end * 0.7 else 0.0

        w2 = 1.0 if self.current_epoch >= self.t0 * 0.7 else 0.0
        # w(t): 从极小值缓慢增大到1，sigmoid形状比线性更平滑
        if self.current_epoch < self.w_start:
            w3 = 0.01                         # 极小初值，不为0避免完全没有信号
        else:
            progress = (self.current_epoch - self.w_start) / (self.w_end - self.w_start)
            progress = min(progress, 1.0)
            # sigmoid 形状，过渡比线性更平滑
            w3 = 0.01 + 0.99 / (1 + math.exp(-10 * (progress - 0.5)))

        return w1, w2, w3
    
    def _build_scheduler(self):
        """
        分两段：
        - [0, t0]:       warmup(5epoch) + cosine 下降到 min_lr
        - [t0, total]:   restart 到 base_lr * restart_lr_ratio，再 cosine 下降到 min_lr
        restart_lr_ratio: t0处重启的lr相对base_lr的比例，默认0.3
        """

        base_lr = self.config.lr
        min_lr = self.config.scheduler.get("min_lr", 1e-5)
        restart_lr_ratio = self.config.scheduler.get("restart_lr_ratio", 0.3)

        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                # 线性warmup
                return epoch / warmup_epochs
            
            if epoch < self.t0:
                # 第一段cosine
                progress = (epoch - warmup_epochs) / (self.t0 - warmup_epochs)
                cosine   = 0.5 * (1 + math.cos(math.pi * progress))
                # 从1降到min_lr/base_lr
                lo = min_lr / base_lr
                return lo + (1 - lo) * cosine
            
            else:
                # 第二段：t0处重启，短warmup(3epoch)后cosine下降
                restart_warmup = self.config.scheduler.get("restart_warmup", 10)
                elapsed  = epoch - self.t0
                peak     = restart_lr_ratio          # 重启峰值比例
                lo       = min_lr / base_lr

                if elapsed < restart_warmup:
                    # 从min_lr线性warmup到peak
                    return lo + (peak - lo) * (elapsed / restart_warmup)
                else:
                    progress = (elapsed - restart_warmup) / (self.epochs - self.t0 - restart_warmup)
                    progress = min(progress, 1.0)
                    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
                    return lo + (peak - lo) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def on_train_begin(self):
        super().on_train_begin()
        if self.model.__class__.__name__ not in self.CISList:
            raise TypeError(f"传入模型应该为{self.CISList}，got {self.model.__class__.__name__}")

    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        for split in self.loaders:
            sampler = self.loaders[split].dataset
            if hasattr(sampler, 'set_epoch'):        
                sampler.set_epoch(self.current_epoch)

    def _model_train_state_set(self):
        if self.current_epoch < 0.7*self.t0:
            self.model.train()
        else:
            self.model.resFNO.train()
            for p in self.model.upFNO.parameters():
                p.requires_grad = False
            
    
    
    def _train_step(self, batch):
        lr_batch, hr_batch = batch
        head_pred, res_pred = self.model(lr_batch)
        head_res = hr_batch - head_pred
        w1, w2, w3 = self._get_loss_weights()
        loss_head = self.criterion_head(head_pred, hr_batch)
        loss_fuse = self.criterion_fuse(head_pred + res_pred, hr_batch)
        loss_res_pixel = self.criterion_pixel_res(res_pred, head_res.detach())
        loss_res, loss_res_pixel_l1, loss_res_freq = self.criterion_res(res_pred, head_pred, hr_batch)
        loss = w1 * loss_head + (1-w2) * loss_fuse + w2 * loss_res_pixel + w3 * loss_res
        
        return {
            'train_loss': loss,
            "head_loss" : loss_head.detach(),
            "fuse_loss" : loss_fuse.detach(),
            "res_loss_pixel_l1": loss_res_pixel_l1.detach(),
            "res_loss_freq" : loss_res_freq.detach(),
            "w1" : w1,
            "w2" : w2,
            "w3" : w3,
        }

    def _eval_step(self, batch):
        lr_batch, hr_batch = batch
        sr_batch = self.patch_inference(self.model, lr_batch, scale=self.model.scale)
        return {
            "val_loss": self.criterion_head(sr_batch, hr_batch).item(),
            "psnr"    : self.psnr(sr_batch, hr_batch).item(),
            "ssim"    : self.ssim(sr_batch, hr_batch).item(),
        }

    def _get_train_metric_keys(self):
        return ["train_loss", "head_loss", "fuse_loss", "res_loss_pixel_l1", "res_loss_freq", "w1", "w2", "w3"]

    def _get_eval_metric_keys(self):
        return ["val_loss", "psnr", "ssim"]
    
    def _save_checkpoint(self):
        eval_metric_key = self.config.get('eval_metric_key', 'val_loss')
        if eval_metric_key == "psnr" or eval_metric_key == "ssim":
            if self.current_epoch == self.start_epoch:
                self.best_eval_metric = -float('inf')
            checkpoint = {
                "epoch"               : self.current_epoch + 1,
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
            if current >= self.best_eval_metric:
                self.best_eval_metric = current
                torch.save(checkpoint, self.save_path / "best.pth")
                self.logger.info(
                    f"  ↑ Best checkpoint saved | {eval_metric_key}: {current:.5f}"
                )
        else:
                super()._save_checkpoint()
        
    def patch_inference(self, model, lr_tensor, patch_size=128, overlap=16, scale=2):
        """
        lr_tensor: (1, 3, H, W), range [0,1]
        """
        B, C, H, W = lr_tensor.shape
        stride = patch_size - overlap
        
        sr_tensor = torch.zeros(B, C, H*scale, W*scale, device=lr_tensor.device)
        count_map = torch.zeros_like(sr_tensor)

        for top in range(0, H, stride):
            for left in range(0, W, stride):
                t = min(top,  H - patch_size)
                l = min(left, W - patch_size)

                patch = lr_tensor[:, :, t:t+patch_size, l:l+patch_size]

                with torch.no_grad():
                    head_pred, res_pred = model(patch)  # ← 解包两个输出
                    sr_patch = head_pred + res_pred      # ← 真实预测

                sr_tensor[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += sr_patch
                count_map[:, :,
                        t*scale:(t+patch_size)*scale,
                        l*scale:(l+patch_size)*scale] += 1

        return sr_tensor / count_map.clamp(min=1)