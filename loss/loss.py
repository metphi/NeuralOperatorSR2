import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqLoss(nn.Module):
    """
    频域损失：分低频/高频区域加权监督
    低频权重小（head已经能学），高频权重大（当前最弱）
    低频区域由矩形域定义（与输入尺寸比例相似）
    """
    def __init__(self, high_freq_weight=2.0, low_freq_weight=1.0, low_freq_ratio=0.1):
        """
        high_freq_weight: 高频区域的权重
        low_freq_weight:  低频区域的权重
        low_freq_ratio:   低频矩形区域的半边长占图像尺寸的比例
        """
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.low_freq_weight  = low_freq_weight
        self.low_freq_ratio   = low_freq_ratio

    def _make_weight_mask(self, H, W, device):
        # FFT 原生坐标：行频率 [0,1,...,H//2, -H//2+1,...,-1]，列同理
        fy = torch.fft.fftfreq(H, device=device) * H   # [-H//2, ..., H//2-1] 重排为 FFT 顺序
        fx = torch.fft.rfftfreq(W, device=device) * W  # rfft: [0, 1, ..., W//2]，共 W//2+1 个

        # 低频矩形半边长
        half_h = (H // 2) * self.low_freq_ratio
        half_w = (W // 2) * self.low_freq_ratio

        in_rect = (fy[:, None].abs() <= half_h) & (fx[None, :].abs() <= half_w)  # [H, W//2+1]

        mask_r = torch.where(
            in_rect,
            torch.full((H, W // 2 + 1), self.low_freq_weight,  device=device),
            torch.full((H, W // 2 + 1), self.high_freq_weight, device=device),
        )
        return mask_r  # [H, W//2+1]，直接对齐 rfft2 输出，无需 fftshift

    def forward(self, pred, target):
        # pred, target: [B, C, H, W]
        B, C, H, W = pred.shape

        # rfft2 更省显存（利用共轭对称性）
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')  # [B, C, H, W//2+1]
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 对实部和虚部分别算 L1（比 L2 对异常频率更鲁棒）
        diff_real = (pred_fft.real - target_fft.real).abs()
        diff_imag = (pred_fft.imag - target_fft.imag).abs()
        diff = diff_real + diff_imag                         # [B, C, H, W//2+1]

        mask_r = self._make_weight_mask(H, W, pred.device)

        weighted = diff * mask_r.unsqueeze(0).unsqueeze(0)   # broadcast 到 [B, C, H, W//2+1]
        return weighted.mean()
    
class ResExplicitLoss(nn.Module):
    """
    对 res_pred 直接用 head_res = gt - head_pred 做显式监督
    同时在像素域 + 频域双重约束
    """
    def __init__(self, freq_weight=0.1, low_freq_ratio=0.4, low_freq_weight=1.0, high_freq_weight=3.0):
        super().__init__()
        # res 学的是高频残差，所以高频权重要更大
        self.freq_loss = FreqLoss(high_freq_weight=high_freq_weight,
                                  low_freq_ratio=low_freq_ratio, low_freq_weight=low_freq_weight)
        self.freq_weight = freq_weight

    def forward(self, res_pred, head_pred, gt):
        # 切断 head_pred 的梯度：让 res 分支专注学残差，不干扰 head
        head_res = (gt - head_pred).detach()

        # 像素域 L1
        pixel_loss = F.l1_loss(res_pred, head_res)

        # 频域损失（高频加权）
        freq_loss = self.freq_loss(res_pred, head_res)

        return (pixel_loss + self.freq_weight * freq_loss, pixel_loss, freq_loss)