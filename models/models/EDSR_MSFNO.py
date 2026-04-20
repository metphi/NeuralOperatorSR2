import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODEL_REGISTRY
from omegaconf import OmegaConf
from models.fundamental import *
from models.utils import *


class EDSREncoder(nn.Module):
    """EDSR特征提取器（去掉上采样部分）"""
    def __init__(self, in_channel=3, out_channel=64, num_blocks=16, res_scale=0.1, global_res_scale=0.1, activation='gelu'):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_first.weight, mean=0, std=0.02)
        if self.conv_first.bias is not None:
            nn.init.zeros_(self.conv_first.bias)
        self.res_scale = res_scale
        self.global_res_scale=global_res_scale
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
        x = self.global_res_scale * x + residual  # 长跳跃连接
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1, activation='gelu'):
        super().__init__()
        
        self.activation = get_activation(activation)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.02)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.02)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        self.res_scale = res_scale
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.res_scale * x + residual
        return x
    
    
    
class _FNOBlocks(nn.Module):
    def __init__(self, num_channel, n_layers, modes1, modes2, init_method = "kaiming", *args, **kwargs):
        super().__init__()
        self.width = num_channel
        self.modes1, self.modes2 = modes1, modes2
        self.padding = 2
        self.n_layers = n_layers
        
        self.convs = nn.ModuleList([
            SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, init_method=init_method)
            for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(n_layers)
        ])
        
        self.mlps = nn.ModuleList([
            MLP(self.width, self.width, self.width)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = F.pad(x, [0, self.padding, 0, self.padding])
        for i in range(self.n_layers):
            x1 = self.mlps[i](self.convs[i](x))
            x2 = self.ws[i](x)
            x = x1 + x2
            if i != self.n_layers - 1:
                x = torch.sin(x)
        x = x[..., :-self.padding, :-self.padding]
        return x


class _FNO(nn.Module):
    def __init__(self, in_channel, out_channel, modes1 = 60, modes2 = 60, width = 32, n_layers = 4, scale = 2,
                 activation='gelu',
                 init_method='kaiming',
                 ):
        super().__init__()
        self.s = scale
        self.activation = get_activation(activation)
        self.Lifting = nn.Sequential(
            nn.Conv2d(in_channel, width, kernel_size=3, padding=1),
        )
        
        self.blocks = _FNOBlocks(width, n_layers, modes1, modes2, init_method)
        self.Projecting = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(2 * width, out_channel * (self.s ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.s),
        )
        
    def forward(self, x):
        x = self.Lifting(x)
        x = self.blocks(x)
        x = self.Projecting(x)
        return x
    

class MultiScaleFNO2d(nn.Module):
    """
    多尺度 Fourier 神经算子（MS-FNO）的 2D 版本，适用于图像输入。
    结构：多个 FNO 分支，每个分支对输入进行不同程度的缩放（通过乘以不同的缩放因子），并在输出层进行加权
    """
    def __init__(
        self,
        in_channel=64,
        out_channel=3,
        modes1      = 60,
        modes2      = 60,
        width       = 32,
        layers    = 4,
        scale       = 2,
        activation  = 'gelu',
        init_method = 'kaiming',
        num_subnets = 8,
        init_scales = None,   # list of floats, length must equal num_subnets
    ):
        super().__init__()
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_subnets = num_subnets

        # Default initial scaling factors (same spread as original MS-FNO paper)
        if init_scales is None:
            init_scales = [1.0, 40.0, 80.0, 100.0, 120.0, 140.0, 180.0, 200.0]
        assert len(init_scales) == num_subnets, (
            f"len(init_scales)={len(init_scales)} must equal num_subnets={num_subnets}"
        )

        self.subnets = nn.ModuleList([
            _FNO(in_channel, out_channel, modes1, modes2, width, layers, scale, activation, init_method)
            for _ in range(num_subnets)
        ])

        # Both are trainable, exactly as in original MS-FNO
        self.scaling_factors = nn.Parameter(torch.tensor(init_scales, dtype=torch.float32))
        self.weights         = nn.Parameter(torch.ones(num_subnets,   dtype=torch.float32))

    def forward(self, x):
        # Convert to channel-last for Linear lifting
        x_cl = x
 
        outputs = []
        for i, net in enumerate(self.subnets):
            scale = self.scaling_factors[i]
 
            # Scale the input features + append spatial grid  (≡ original MS-FNO)
            x_scaled = scale * x_cl                      
            x_in = x_scaled                                    # [B, in_channel, H, W]
            # print(x_in.shape)
            out = net(x_in) 
            outputs.append(out)
 
        # Weighted sum → [B,  out_channel, H, W]
        stacked = torch.stack(outputs, dim=0)                           # [num_subnets, C, B, H, W]
        result  = torch.einsum('i,i...->...', self.weights, stacked)    # [B, out_channel, H, W]
        return result                          # [B, out_channel, H, W]
    
    
@MODEL_REGISTRY.register("EDSR_MSFNO")
class EDSR_MSFNO(nn.Module):
    def __init__(self, 
                 config,
                 *args, **kwargs):
        super().__init__()
        
        cfg = config.model
        cfg_ms = OmegaConf.to_container(cfg["MSFNO"], resolve=True)
        cfg_encoder = OmegaConf.to_container(cfg["Encoder"], resolve=True)
        self.scale = cfg_ms['scale']
        self.encoder = EDSREncoder(**cfg_encoder)
        self.msFNO = MultiScaleFNO2d(**cfg_ms)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False,)
        fea = self.encoder(x)
        x   = self.msFNO(fea)
        x = x + base
        return x
