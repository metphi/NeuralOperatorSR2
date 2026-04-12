import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODEL_REGISTRY
from omegaconf import OmegaConf
from models.fundamental import *
from models.utils import *


class ResFNO(nn.Module):
    def __init__(self, in_channel, out_channel, modes1 = 60, modes2 = 60, width = 32, n_layers = 4, 
                 activation='gelu',
                 init_method='kaiming',
                 GroupNorm = True,):
        super().__init__()
        self.activation = get_activation(activation)
        self.Lifting = nn.Sequential(
            nn.Conv2d(in_channel, width, kernel_size=3, padding=1,),
            self.activation,
            nn.Conv2d(width, width, kernel_size=3, padding=1)
        )
        self.blocks = FNOBlocks(width, n_layers, modes1, modes2, init_method, activation)
        self.GroupNorm = GroupNorm
        self.GN = nn.GroupNorm(4, width)
        self.Projecting = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(width, out_channel, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.Lifting(x)
        x = self.blocks(x)
        if self.GroupNorm:
            x = self.GN(x)
        x = self.Projecting(x)
        return x
        
    
class UpSampleFNO(nn.Module):
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
        base = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False,)
        x = self.Lifting(x)
        x = self.blocks(x)
        if self.GroupNorm:
            x = self.GN(x)
        x = self.Projecting(x)
        x = x + base
        return x.clamp(0.0,1.0), x

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
        return x.clamp(0.0,1.0), x


class _FNO2d(nn.Module):
    """
    One subnet of MultiScaleFNO2d.
 
    Input  : [B, H, W, in_channel + 2]   (channel-last, grid coords appended)
    Output : [B, H, W, out_channel]       (channel-last)
    """
    def __init__(self, in_channel, out_channel, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width
 
        # Lifting: (in_channel + 2) → width  [same as original MS-FNO p layer]
        self.p = nn.Linear(in_channel + 2, width)
 
        # 4 Fourier layers
        self.conv0 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_fast(width, width, modes1, modes2)
 
        self.mlp0 = MLP(width, width, width)
        self.mlp1 = MLP(width, width, width)
        self.mlp2 = MLP(width, width, width)
        self.mlp3 = MLP(width, width, width)
 
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
 
        # Projection: width → out_channel
        self.q = MLP(width, out_channel, width * 4)
 
    def forward(self, x):
        # x : [B, H, W, in_channel + 2]
        x = self.p(x)               # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
 
        x1 = self.conv0(x); x1 = self.mlp0(x1); x2 = self.w0(x)
        x  = torch.sin(x1 + x2)
 
        x1 = self.conv1(x); x1 = self.mlp1(x1); x2 = self.w1(x)
        x  = torch.sin(x1 + x2)
 
        x1 = self.conv2(x); x1 = self.mlp2(x1); x2 = self.w2(x)
        x  = torch.sin(x1 + x2)
 
        x1 = self.conv3(x); x1 = self.mlp3(x1); x2 = self.w3(x)
        x  = x1 + x2                # last layer: no activation
 
        x = self.q(x)               # [B, out_channel, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, out_channel]
        return x

class _FNO2d_noW(nn.Module):

    def __init__(self, in_channel, out_channel, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width
 
        # Lifting: (in_channel + 2) → width  [same as original MS-FNO p layer]
        self.p = nn.Linear(in_channel + 2, width)
 
        # 4 Fourier layers
        self.conv0 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_fast(width, width, modes1, modes2)
 
        self.mlp0 = MLP(width, width, width)
        self.mlp1 = MLP(width, width, width)
        self.mlp2 = MLP(width, width, width)
        self.mlp3 = MLP(width, width, width)
 
        # Projection: width → out_channel
        self.q = MLP(width, out_channel, width * 4)
 
    def forward(self, x):
        # x : [B, H, W, in_channel + 2]
        x = self.p(x)               # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
 
        x1 = self.conv0(x); x1 = self.mlp0(x1)
        x  = torch.sin(x1)
 
        x1 = self.conv1(x); x1 = self.mlp1(x1)
        x  = torch.sin(x1)
 
        x1 = self.conv2(x); x1 = self.mlp2(x1)
        x  = torch.sin(x1)
 
        x1 = self.conv3(x); x1 = self.mlp3(x1)
        x  = x1                       # last layer: no activation
 
        x = self.q(x)               # [B, out_channel, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, out_channel]
        return x




class MultiScaleFNO2d(nn.Module):
    """
    Multi-scale FNO for high-frequency residual learning.
 
    Replaces ResFNO in CISFNO.  Accepts a standard image-format tensor
    [B, in_channel, H, W] and returns [B, out_channel, H, W].
 
    Each of the `num_subnets` parallel subnets receives the input rescaled by
    a (trainable) scaling factor together with normalised x/y grid coordinates,
    mirroring the original Multi-Scale FNO design exactly.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        modes1      = 60,
        modes2      = 60,
        width       = 32,
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
            _FNO2d(in_channel, out_channel, modes1, modes2, width)
            for _ in range(num_subnets)
        ])
 
        # Both are trainable, exactly as in original MS-FNO
        self.scaling_factors = nn.Parameter(torch.tensor(init_scales, dtype=torch.float32))
        self.weights         = nn.Parameter(torch.ones(num_subnets,   dtype=torch.float32))
 
    # ------------------------------------------------------------------
    def _get_scaled_grid(self, shape, device, scale):
        """
        Returns a grid tensor [B, H, W, 2] with coordinates in
        [0, scale] × [0, scale], matching the original MS-FNO get_scaled_grid.
        """
        B, _, H, W = shape
        gx = torch.linspace(0, 1, H, device=device) * scale   # [H]
        gy = torch.linspace(0, 1, W, device=device) * scale   # [W]
        gx = gx.view(1, H, 1, 1).expand(B, H, W, 1)
        gy = gy.view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)                     # [B, H, W, 2]
 
    # ------------------------------------------------------------------
    def forward(self, x):
        # x: [B, in_channel, H, W]
        B, C, H, W = x.shape
 
        # Convert to channel-last for Linear lifting
        x_cl = x.permute(0, 2, 3, 1)   # [B, H, W, in_channel]
 
        outputs = []
        for i, net in enumerate(self.subnets):
            scale = self.scaling_factors[i]
 
            # Scale the input features + append spatial grid  (≡ original MS-FNO)
            x_scaled = scale * x_cl                                     # [B, H, W, in_channel]
            grid     = self._get_scaled_grid(x.shape, x.device, scale)  # [B, H, W, 2]
            x_in     = torch.cat([x_scaled, grid], dim=-1)              # [B, H, W, in_channel+2]
 
            out = net(x_in)             # [B, H, W, out_channel]
            outputs.append(out)
 
        # Weighted sum → [B, H, W, out_channel]
        stacked = torch.stack(outputs, dim=0)                           # [num_subnets, B, H, W, C]
        result  = torch.einsum('i,i...->...', self.weights, stacked)    # [B, H, W, out_channel]
 
        return result.permute(0, 3, 1, 2)                               # [B, out_channel, H, W]


class MultiScaleFNO2d_noW(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        modes1      = 60,
        modes2      = 60,
        width       = 32,
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
            _FNO2d_noW(in_channel, out_channel, modes1, modes2, width)
            for _ in range(num_subnets)
        ])
 
        # Both are trainable, exactly as in original MS-FNO
        self.scaling_factors = nn.Parameter(torch.tensor(init_scales, dtype=torch.float32))
        self.weights         = nn.Parameter(torch.ones(num_subnets,   dtype=torch.float32))
 
    # ------------------------------------------------------------------
    def _get_scaled_grid(self, shape, device, scale):
        """
        Returns a grid tensor [B, H, W, 2] with coordinates in
        [0, scale] × [0, scale], matching the original MS-FNO get_scaled_grid.
        """
        B, _, H, W = shape
        gx = torch.linspace(0, 1, H, device=device) * scale   # [H]
        gy = torch.linspace(0, 1, W, device=device) * scale   # [W]
        gx = gx.view(1, H, 1, 1).expand(B, H, W, 1)
        gy = gy.view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)                     # [B, H, W, 2]
 
    # ------------------------------------------------------------------
    def forward(self, x):
        # x: [B, in_channel, H, W]
        B, C, H, W = x.shape
 
        # Convert to channel-last for Linear lifting
        x_cl = x.permute(0, 2, 3, 1)   # [B, H, W, in_channel]
 
        outputs = []
        for i, net in enumerate(self.subnets):
            scale = self.scaling_factors[i]
 
            # Scale the input features + append spatial grid  (≡ original MS-FNO)
            x_scaled = scale * x_cl                                     # [B, H, W, in_channel]
            grid     = self._get_scaled_grid(x.shape, x.device, scale)  # [B, H, W, 2]
            x_in     = torch.cat([x_scaled, grid], dim=-1)              # [B, H, W, in_channel+2]
 
            out = net(x_in)             # [B, H, W, out_channel]
            outputs.append(out)
 
        # Weighted sum → [B, H, W, out_channel]
        stacked = torch.stack(outputs, dim=0)                           # [num_subnets, B, H, W, C]
        result  = torch.einsum('i,i...->...', self.weights, stacked)    # [B, H, W, out_channel]
 
        return result.permute(0, 3, 1, 2)                               # [B, out_channel, H, W]






@MODEL_REGISTRY.register("CISFNO")
class CISFNO(nn.Module):
    def __init__(self, 
                 config,
                 *args, **kwargs):
        super().__init__()
        
        cfg = config.model
        cfg_up = OmegaConf.to_container(cfg["UpSampleFNO"], resolve=True)
        cfg_res = OmegaConf.to_container(cfg["ResFNO"], resolve=True)
        self.scale = cfg_up['scale']
        self.upFNO = UpSampleFNO(**cfg_up)
        self.resFNO = MultiScaleFNO2d(**cfg_res)
        
    def forward(self, x):
        c_x, x = self.upFNO(x)
        res = self.resFNO(x)
        
        return c_x, res
    
    
    
@MODEL_REGISTRY.register("CISFNO2")
class CISFNO2(nn.Module):
    def __init__(self, 
                 config,
                 *args, **kwargs):
        super().__init__()
        
        cfg = config.model
        cfg_up = OmegaConf.to_container(cfg["UpSampleFNO"], resolve=True)
        cfg_res = OmegaConf.to_container(cfg["ResFNO"], resolve=True)
        self.scale = cfg_up['scale']
        self.upFNO = UpSampleFNO(**cfg_up)
        self.resFNO = MultiScaleFNO2d_noW(**cfg_res)
        
    def forward(self, x):
        c_x, x = self.upFNO(x)
        res = self.resFNO(x)
        
        return c_x, res
    
    
@MODEL_REGISTRY.register("CISFNO3")
class CISFNO(nn.Module):
    def __init__(self, 
                 config,
                 *args, **kwargs):
        super().__init__()
        
        cfg = config.model
        cfg_up = OmegaConf.to_container(cfg["UpSampleFNO"], resolve=True)
        cfg_res = OmegaConf.to_container(cfg["ResFNO"], resolve=True)
        self.scale = cfg_up['scale']
        self.upFNO = UpSampleFNO2(**cfg_up)
        self.resFNO = MultiScaleFNO2d(**cfg_res)
        
    def forward(self, x):
        c_x, x = self.upFNO(x)
        res = self.resFNO(x)
        
        return c_x, res