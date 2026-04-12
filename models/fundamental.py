import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Point-wise MLP implemented as 1×1 convolutions (operates on [B,C,H,W])."""
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels,  mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
 
    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_method='kaiming'):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        if init_method == 'kaiming':
            self.weights1 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
            self.weights2 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        elif init_method == 'zeros':
            self.weights1 = nn.Parameter(
                torch.zeros(
                    in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
            self.weights2 = nn.Parameter(
                torch.zeros(
                    in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
            

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
    
    
class FNOBlocks(nn.Module):
    def __init__(self, num_channel, n_layers, modes1, modes2, init_method = "kaiming", activation = "gelu", *args, **kwargs):
        super().__init__()
        self.width = num_channel
        self.modes1, self.modes2 = modes1, modes2
        self.padding = 2
        self.n_layers = n_layers
        
        if activation is None:
            self.activation = None
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.convs = nn.ModuleList([
            SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, init_method=init_method)
            for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        x = F.pad(x, [0, self.padding, 0, self.padding])
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i != self.n_layers - 1 and self.activation is not None:
                x = self.activation(x)
        x = x[..., :-self.padding, :-self.padding]
        return x