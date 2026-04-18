import math
import torch
from utils._utils import *
from utils.merics_utlis import SRMetric
from loss.loss import FreqLoss, ResExplicitLoss
from trainer.base_trainer import BaseTrainer
from trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("CIPtrainer")
class CIPtrainer(BaseTrainer):

    def __init__(self, model, loaders, config):
        # self.CIPList = ["CIPFNO", "CIPModel", "CIPFNO1"]     
        super().__init__(model, loaders, config)                          

    def _setup(self):
        super()._setup()
        norm = self._ALL_CONFIG.data.get("norm", None)
        if norm is None:
            self.norm = {'lr': {'sub': 0, 'div': 1}, 'gt': {'sub': 0, 'div': 1}}
        else:
            self.norm = norm
        self.alpha     = self.config.get("alpha", 1.0)
        scale = self._ALL_CONFIG.data.get("scale", 2)
        self.psnr_y      = SRMetric('psnr', test_y_channel=True, crop_border=scale+6)
        self.ssim_y      = SRMetric('ssim', test_y_channel=True, crop_border=scale+6)
        self.psnr_rgb    = SRMetric('psnr', test_y_channel=False, crop_border=scale+6)
        self.ssim_rgb    = SRMetric('ssim', test_y_channel=False, crop_border=scale+6)
        self.criterion_skip = nn.MSELoss()
        self.criterion_up = nn.L1Loss()

    def on_train_begin(self):
        super().on_train_begin()
        # if self.model.__class__.__name__ not in self.CIPList:
        #     raise TypeError(f"传入模型应该为{self.CIPList}，got {self.model.__class__.__name__}")

    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        for split in self.loaders:
            sampler = self.loaders[split].dataset
            if hasattr(sampler, 'set_epoch'):        
                sampler.set_epoch(self.current_epoch)

    def _train_epoch(self):
        self._model_train_state_set()
        train_keys = self._get_train_metric_keys()
        backward_key = train_keys[0]
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.loaders['train']):
            batch_size = batch[0].shape[0]
            batch = [item.to(self.device) for item in batch]
            self.optimizer.zero_grad()
            out = self._train_step(batch)
            
            # backward
            out[backward_key].backward()
            
            # ========== 局部梯度监控 ==========
            if hasattr(self.model, "encoder"):
                total_norm = sum(p.grad.data.norm(2).item() ** 2 
                                for p in self.model.encoder.parameters() 
                                if p.grad is not None) ** 0.5
                
                # 如果梯度过大,打印
                if total_norm > 15.0:  # 阈值可调
                    self.logger.warning(
                        f"⚠️ Batch {batch_idx}: Encoder Grad Norm = {total_norm:.4f}, "
                        f"Loss = {out[backward_key].item():.6f}"
                    )

                    if self.config.get("encoder_grad_clip", None):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.encoder.parameters(), self.config.encoder_grad_clip
                        )
                        self.logger.info(f"{self.current_epoch}: {batch_idx}: Encoder Grad has been clipped to ±{self.config.encoder_grad_clip}")
                
            # 梯度裁剪
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

    def _train_step(self, batch):
        lr_batch, hr_batch = batch
        skip_pred, up_pred = self.model(lr_batch)
        skip_res = hr_batch - skip_pred
        loss1  = self.criterion_skip(skip_pred, hr_batch)
        loss2  = self.criterion_up(up_pred + skip_pred, hr_batch)
        loss   = loss1 + self.alpha * loss2
        return {
            'train_loss': loss,
            "skip_loss" : loss1.detach(),
            "up_loss"  : loss2.detach(),
        }

    def _eval_step(self, batch):
        lr_batch, hr_batch = batch
        sr_batch = self.patch_inference(self.model, lr_batch, scale=self.model.scale)
        hr_batch = hr_batch * self.norm['gt']['div'] + self.norm['gt']['sub']
        sr_batch = (sr_batch * self.norm['gt']['div'] + self.norm['gt']['sub']).clamp(0, 1)
        return {
            "val_loss": self.criterion_skip(sr_batch, hr_batch).item(),
            "psnr_y"    : self.psnr_y(sr_batch, hr_batch).item(),
            "ssim_y"    : self.ssim_y(sr_batch, hr_batch).item(),
            "psnr_rgb"  : self.psnr_rgb(sr_batch, hr_batch).item(),
            "ssim_rgb"  : self.ssim_rgb(sr_batch, hr_batch).item(),
        }

    def _get_train_metric_keys(self):
        return ["train_loss", "skip_loss", "up_loss"]

    def _get_eval_metric_keys(self):
        return ["val_loss", "psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb"]

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
            if (self.current_epoch+1) % 20 == 0:
                torch.save(checkpoint, self.save_path / "last.pth")
            # 按主指标保存 best
            if self.current_epoch >= 0.75*self.config.epochs:
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