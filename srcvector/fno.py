import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes1 = min(self.modes1, height)
        modes2 = min(self.modes2, width // 2 + 1)
        if modes1 > 0 and modes2 > 0:
            out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
                x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
            )
            out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
                x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
            )

        x = torch.fft.irfft2(out_ft, s=(height, width))
        return x


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        x = self.norm(x)
        return F.gelu(x)


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 48,
        n_layers: int = 4,
        smooth_mode: str = 'none',
        smooth_kernel_size: int = 3,
        final_conv_kernel_size: int = 1,
    ):
        super().__init__()
        assert smooth_mode in ['none', 'gaussian_fixed', 'gaussian']
        self.smooth_mode = smooth_mode

        self.input_proj = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                width * 2,
                out_channels,
                kernel_size=final_conv_kernel_size,
                padding=(final_conv_kernel_size - 1) // 2,
                padding_mode='circular',
            ),
        )

        if smooth_mode != 'none':
            oneD_kernel = cv2.getGaussianKernel(smooth_kernel_size, sigma=0)
            twoD_kernel = (oneD_kernel * oneD_kernel.T).astype(np.float32)
            weight = torch.tensor(twoD_kernel).view(1, 1, smooth_kernel_size, smooth_kernel_size)
            weight = weight.repeat(out_channels, 1, 1, 1)
            self.smooth = nn.Conv2d(
                out_channels,
                out_channels,
                padding=(smooth_kernel_size - 1) // 2,
                padding_mode='circular',
                kernel_size=smooth_kernel_size,
                bias=False,
                groups=out_channels,
            )
            self.smooth.weight = nn.Parameter(weight, requires_grad=(smooth_mode == 'gaussian'))

    def _grid(self, x):
        batch_size, _, height, width = x.shape
        grid_y = torch.linspace(0, 1, steps=height, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0, 1, steps=width, device=x.device, dtype=x.dtype)
        grid_y = grid_y.view(1, 1, height, 1).repeat(batch_size, 1, 1, width)
        grid_x = grid_x.view(1, 1, 1, width).repeat(batch_size, 1, height, 1)
        return torch.cat([grid_y, grid_x], dim=1)

    def forward(self, x):
        x = torch.cat([x, self._grid(x)], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.decoder(x)
        if self.smooth_mode != 'none':
            x = self.smooth(x)
        return x
