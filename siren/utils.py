"""Image IO, metrics and coordinate helpers shared by the demo scripts."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def read_image(path: str) -> np.ndarray:
    """-> (H, W, 3) float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def write_image(path: str, img) -> None:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, over the whole tensor."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(torch.as_tensor(max_val) / torch.sqrt(mse))).item()


def pixel_centers(h: int, w: int, device) -> torch.Tensor:
    """(h*w, 2) coordinates of pixel centres in [0, 1]^2, row-major (y, x).

    Column 0 is x (width), column 1 is y (height), so the result can be
    reshaped to (h, w, ...) directly.
    """
    ys = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
    xs = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xv.reshape(-1), yv.reshape(-1)), dim=1)


class BilinearImage(torch.nn.Module):
    """Bilinearly filtered lookup into a reference image, for random-sample training."""

    def __init__(self, data: np.ndarray, device):
        super().__init__()
        self.register_buffer("data", torch.from_numpy(data).float().to(device))
        self.h, self.w = data.shape[:2]

    @torch.no_grad()
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (B, 2) in [0, 1]^2, column 0 = x -> (B, C)."""
        # MINUS A HALF PIXEL. Pixel j covers [j/w, (j+1)/w) and its centre is at
        # (j+0.5)/w, so without this the lookup treats the pixel as living at its
        # top-left corner: a coordinate at the centre of pixel j lands halfway
        # between j and j+1 and returns their average. The whole reference is
        # then a half-pixel box blur of the image, the fit learns that, and the
        # PSNR is quoted against it -- measured 45.91 dB against the blurred
        # target and 31.18 dB against the actual pixels, for the same fit.
        # deform.sample_bilinear has always had the -0.5; this did not.
        size = torch.tensor([self.w, self.h], device=xy.device, dtype=xy.dtype)
        p = xy * size - 0.5
        # Clamped BEFORE the floor, so the half-pixel band outside the centre
        # grid returns the border pixel instead of interpolating away from it.
        p = torch.stack([p[:, 0].clamp(0, self.w - 1),
                         p[:, 1].clamp(0, self.h - 1)], dim=1)
        i = torch.floor(p)
        f = p - i
        i = i.long()
        x0 = i[:, 0].clamp(0, self.w - 1)
        y0 = i[:, 1].clamp(0, self.h - 1)
        x1 = (x0 + 1).clamp(max=self.w - 1)
        y1 = (y0 + 1).clamp(max=self.h - 1)
        fx, fy = f[:, 0:1], f[:, 1:2]
        return (
            self.data[y0, x0] * (1 - fx) * (1 - fy)
            + self.data[y0, x1] * fx * (1 - fy)
            + self.data[y1, x0] * (1 - fx) * fy
            + self.data[y1, x1] * fx * fy
        )


@torch.no_grad()
def render(model, coords: torch.Tensor, shape, chunk: int = 1 << 20) -> torch.Tensor:
    """Evaluate `model` over `coords` in chunks and reshape to `shape`."""
    out = [model(coords[i : i + chunk]) for i in range(0, coords.shape[0], chunk)]
    return torch.cat(out, dim=0).reshape(shape)
