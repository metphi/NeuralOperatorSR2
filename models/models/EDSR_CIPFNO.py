import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODEL_REGISTRY
from omegaconf import OmegaConf
from models.fundamental import *
from models.utils import *
from CIPFNO import MultiScaleFNO2d, UpSampleFNO2




class EDSREncoder(nn.Module):
    """EDSR特征提取器（去掉上采样部分）"""
    def __init__(self, in_channels=3, out_channels=64, num_blocks=16, res_scale=0.1, activation='gelu'):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels, res_scale, activation) 
            for _ in range(num_blocks)
        ])
        
        # 特征融合（长跳跃连接）
        self.conv_last = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
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
        x = x.mul(self.res_scale) + residual
        return x


    
@MODEL_REGISTRY.register("EDSR-CIPFNO")
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
        self.skip_FNO = UpSampleFNO2(**cfg_skip)
        self.up_FNO = MultiScaleFNO2d(**cfg_up)

    def forward(self, x):
        x = self.encoder(x)
        x_skip = self.skip_FNO(x)
        x_up   = self.up_FNO(x)
        return x_skip, x_up