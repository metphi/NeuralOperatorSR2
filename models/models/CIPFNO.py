import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODEL_REGISTRY
from omegaconf import OmegaConf
from models.fundamental import *
from models.utils import *



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
        in_channel,
        out_channel,
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
    
    
class UpSampleFNO2(nn.Module):
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
        base = F.interpolate(x, scale_factor=self.s, mode='bilinear', align_corners=False,)
        x = self.Lifting(x)
        x = self.blocks(x)
        if self.GroupNorm:
            x = self.GN(x)
        x = self.Projecting(x)
        x = x + base
        return x
    
    
@MODEL_REGISTRY.register("CIPFNO")
class CIPFNO(nn.Module):
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
        self.skip_FNO = UpSampleFNO2(**cfg_skip)
        self.up_FNO = MultiScaleFNO2d(**cfg_up)

    def forward(self, x):
        x_skip = self.skip_FNO(x)
        x_up   = self.up_FNO(x)
        return x_skip, x_up