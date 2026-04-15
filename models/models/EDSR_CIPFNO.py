import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODEL_REGISTRY
from omegaconf import OmegaConf
from models.fundamental import *
from models.utils import *
from models.models.CIPFNO import MultiScaleFNO2d


class _UpSampleFNO2(nn.Module):
    def __init__(self, in_channel, out_channel, modes1 = 4, modes2 = 4, width = 32, n_layers = 4, scale = 2,
                 activation='gelu',
                 init_method='kaiming',
                 GroupNorm = True,
                 ):
        super().__init__()
        self.s = scale
        self.activation = get_activation(activation)
        self.Lifting = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(in_channel, width, kernel_size=3, padding=1)
        )
        
        self.blocks = FNOBlocks(width, n_layers, modes1, modes2, init_method, activation)
        
        self.GroupNorm = GroupNorm
        self.GN = nn.GroupNorm(4, width)
        self.Projecting = nn.Sequential(
            nn.Conv2d(width, out_channel *self.s**2, kernel_size=3, padding=1),
            nn.PixelShuffle(self.s),
            self.activation,
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.Lifting(x)
        x = self.blocks(x)
        if self.GroupNorm:
            x = self.GN(x)
        x = self.Projecting(x)
        return x

class EDSREncoder(nn.Module):
    """EDSR特征提取器（去掉上采样部分）"""
    def __init__(self, in_channel=3, out_channel=64, num_blocks=16, res_scale=0.1, activation='gelu'):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        
        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channel, res_scale, activation) 
            for _ in range(num_blocks)
        ])
        
        # 特征融合（长跳跃连接）
        self.conv_last = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_first(x)
        residual = x
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.conv_last(x)
        x = x + residual  # 长跳跃连接
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1, activation='gelu'):
        super().__init__()
        
        self.activation = get_activation(activation)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.res_scale = res_scale
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.res_scale * x + residual
        return x


    
@MODEL_REGISTRY.register("EDSR_CIPFNO")
class EDSR_CIPFNO(nn.Module):
    def __init__(self, 
                 config,
                 *args, **kwargs):
        super().__init__()
        
        cfg = config.model
        s1 = cfg.SkipFNO.scale
        s2 = cfg.UpFNO.scale
        assert s1 == s2, "skip_FNO 和 up_FNO 的 scale 必须相同"
        self.scale = s1
        cfg_skip = OmegaConf.to_container(cfg["SkipFNO"], resolve=True)
        cfg_up = OmegaConf.to_container(cfg["UpFNO"], resolve=True)
        cfg_encoder = OmegaConf.to_container(cfg["Encoder"], resolve=True)
        self.encoder = EDSREncoder(**cfg_encoder)
        self.skip_FNO = _UpSampleFNO2(**cfg_skip)
        self.up_FNO = MultiScaleFNO2d(**cfg_up)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False,)
        fea = self.encoder(x)
        x_skip = self.skip_FNO(fea) + base
        x_up   = self.up_FNO(fea)
        return x_skip, x_up