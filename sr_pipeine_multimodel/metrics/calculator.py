import lpips
import torch
from pytorch_fid import fid_score
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class MetricCalculator:
    def __init__(self, device):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.lpips = lpips.LPIPS(net="vgg").to(device)

    def __call__(self, fake, real):
        metrics = {
            "MSE": torch.mean((fake - real) ** 2).item(),
            "MAE": torch.mean(torch.abs(fake - real)).item(),
            "SSIM": self.ssim(fake, real).item(),
            "PSNR": self.psnr(fake, real).item(),
            "LPIPS": self.lpips(fake, real).mean().item(),
        }
        return metrics


def calculate_fid(real_dir, fake_dir):
    return fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
