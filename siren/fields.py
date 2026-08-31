"""Analytic time-evolving 2D field used as ground truth for the (x, y, t) demo.

A sum of Fourier modes drifting at constant velocity c and decaying at rate
nu|k|^2 -- the exact solution of the advection-diffusion equation

    du/dt + c . grad u = nu laplacian u                                    (*)

on the periodic unit square.  Two things follow, and both are used by the
demos:

  * u is known in closed form at *any* (x, y, t), so the error of a fit at
    times that were never trained on is an exact number, not an estimate
    against a nearest stored frame;
  * u's first and second derivatives are known in closed form too, so the
    derivatives the NGP produces by autograd can be checked against truth,
    and the fitted field can be scored by how well it satisfies (*).
"""

from __future__ import annotations

import math

import torch


class AdvDiffField:
    """u(x, y, t) -> [0, 1]-ish scalar, exact solution of advection-diffusion.

    Args:
        n_modes: number of Fourier modes summed.
        k_max: largest wavenumber (in cycles per domain) along each axis.
        nu: diffusivity; nu*(2*pi*k_max)^2 sets how fast the finest structure
            fades over t in [0, 1].
        velocity: drift c, in domain-fractions per unit time.
        seed: mode amplitudes/phases/wavevectors are drawn once from this.
    """

    def __init__(
        self,
        n_modes: int = 48,
        k_max: int = 12,
        nu: float = 5.0e-4,
        velocity: tuple[float, float] = (0.35, -0.20),
        seed: int = 0,
        device="cpu",
    ):
        g = torch.Generator().manual_seed(seed)
        # Integer wavevectors, no (0,0), |k|_inf <= k_max.
        k = torch.randint(-k_max, k_max + 1, (4 * n_modes, 2), generator=g)
        k = k[(k.abs().sum(1) > 0)][:n_modes]
        if k.shape[0] < n_modes:
            raise RuntimeError("could not draw enough non-zero wavevectors")
        knorm = k.float().norm(dim=1, keepdim=True)
        amp = torch.rand(n_modes, 1, generator=g).add(0.25) / knorm  # ~1/|k| spectrum
        phase = torch.rand(n_modes, 1, generator=g) * (2 * math.pi)

        self.k = (2 * math.pi * k.float()).to(device)          # (M, 2), angular
        self._amp = amp.to(device).squeeze(1)                  # (M,)
        self.phase = phase.to(device).squeeze(1)               # (M,)
        self.decay = (nu * (self.k**2).sum(1)).to(device)      # (M,) nu|k|^2
        self.c = torch.tensor(velocity, device=device)         # (2,)
        self.nu = nu
        self.device = device

        # Normalise to about [0.05, 0.95] using the t=0 extremes on a fine grid.
        self.scale, self.offset = 1.0, 0.0
        with torch.no_grad():
            probe = _unit_grid(256, device)
            xyt = torch.cat([probe, torch.zeros_like(probe[:, :1])], dim=1)
            v = self(xyt)
            self.scale = float(0.9 / (v.max() - v.min()).clamp(min=1e-6))
            self.offset = float(0.5 - self.scale * 0.5 * (v.max() + v.min()))

    def _phi(self, xyt: torch.Tensor):
        """Per-mode phase and envelope. xyt: (B, 3) -> (B, M), (B, M)."""
        xy, t = xyt[:, :2], xyt[:, 2:3]
        shifted = xy - t * self.c                       # advection: x - c t
        theta = shifted @ self.k.T + self.phase         # (B, M)
        env = torch.exp(-self.decay * t)                # (B, M)
        return theta, env

    def __call__(self, xyt: torch.Tensor) -> torch.Tensor:
        """(B, 3) -> (B, 1)."""
        theta, env = self._phi(xyt)
        u = (self._amp * env * torch.cos(theta)).sum(1, keepdim=True)
        return self.scale * u + self.offset

    def grad(self, xyt: torch.Tensor) -> torch.Tensor:
        """Analytic (du/dx, du/dy, du/dt). (B, 3) -> (B, 3)."""
        theta, env = self._phi(xyt)
        a = self._amp * env                                    # (B, M)
        s = -a * torch.sin(theta)                              # d/dtheta
        dxy = s @ self.k                                       # (B, 2)
        # d/dt gets both the advection of theta and the decay of the envelope.
        dtheta_dt = -(self.k @ self.c)                         # (M,)
        dt = (s * dtheta_dt - a * self.decay * torch.cos(theta)).sum(1, keepdim=True)
        return self.scale * torch.cat([dxy, dt], dim=1)

    def laplacian(self, xyt: torch.Tensor) -> torch.Tensor:
        """Analytic d2u/dx2 + d2u/dy2. (B, 3) -> (B, 1)."""
        theta, env = self._phi(xyt)
        k2 = (self.k**2).sum(1)                                # (M,)
        u = -(self._amp * env * torch.cos(theta) * k2).sum(1, keepdim=True)
        return self.scale * u

    @torch.no_grad()
    def frames(self, ts, res: int = 256) -> torch.Tensor:
        """(T, res, res) ground-truth images at the given times."""
        xy = _unit_grid(res, self.device)
        out = []
        for t in ts:
            xyt = torch.cat([xy, torch.full_like(xy[:, :1], float(t))], dim=1)
            out.append(self(xyt).reshape(res, res))
        return torch.stack(out)


def _unit_grid(res: int, device) -> torch.Tensor:
    """(res*res, 2) pixel-centre coordinates in [0, 1]^2, row-major (y, x)."""
    c = torch.linspace(0.5 / res, 1 - 0.5 / res, res, device=device)
    yv, xv = torch.meshgrid(c, c, indexing="ij")
    return torch.stack((xv.reshape(-1), yv.reshape(-1)), dim=1)
