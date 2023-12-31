import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 4, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:
        self.gaussian_kernel = self.gaussian_kernel.to(x.device) if self.gaussian_kernel.device != x.device else self.gaussian_kernel
        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1. - ssim_map.mean()
        else:
            return torch.mean(1. - ssim_map, dim=-3, keepdim=True)

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float, requires_grad=False)
        kernel_1d = torch.exp(- torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d
