import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .msgms import MSGMSLoss
from .ssim import SSIMLoss


class RIADLoss(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        num_scales: int = 3,
        in_channels: int = 3,
        alpha: float = 0.01,
        beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss_fn = nn.MSELoss(reduction="none")
        self.ssim_loss_fn = SSIMLoss(kernel_size=kernel_size, sigma=sigma)
        self.msgms_loss_fn = MSGMSLoss(num_scales=num_scales, in_channels=in_channels)

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        mse_loss = self.mse_loss_fn(pred, true).mean()
        ssim_loss = self.ssim_loss_fn(pred, true, as_loss=True)
        msgms_loss = self.msgms_loss_fn(pred, true, as_loss=True)
        return mse_loss + self.alpha * ssim_loss + self.beta * msgms_loss


class RIADScore(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        num_scales: int = 3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.msgms_loss_fn = MSGMSLoss(num_scales=num_scales, in_channels=in_channels)
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, requires_grad=False) / kernel_size ** 2
        self.mean_kernel = mean_kernel
        self.kernel_size = kernel_size

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        normal_score = self.msgms_loss_fn(pred, true, as_loss=False)  # similarity; normal score
        anomalous_score = 1. - normal_score

        mean_kernel = self.mean_kernel
        mean_kernel = mean_kernel.to(anomalous_score.device) if mean_kernel.device != anomalous_score.device else mean_kernel
        smoothed_score = F.conv2d(anomalous_score, mean_kernel, padding=self.kernel_size // 2, groups=1)
        smoothed_score = smoothed_score.reshape(smoothed_score.shape[0], -1)
        return smoothed_score.max(dim=1).values



__all__ = [
    "RIADLoss", "RIADScore",
    "MSGMSLoss",
    "SSIMLoss",
]
