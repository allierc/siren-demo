"""Deformation fields: the ground-truth ones, and the two models that fit them.

Everything speaks the same convention:

    target(x) = source(x + u(x)),      x in [0, 1]^2,  u in pixels

so a model is any callable `(N, 2) -> (N, 2)` returning a pixel displacement.
The ground-truth fields are analytic, which is what makes the endpoint error of
a fit a real number rather than a comparison against another fit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .siren import SirenField


# --------------------------------------------------------------- sampling


def sample_bilinear(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """image: (H, W) or (H, W, C); xy: (N, 2) in [0, 1]^2 -> (N, C).

    Differentiable w.r.t. `xy`, which is what the registration loss needs.
    Out-of-range coordinates clamp to the border.
    """
    single = image.dim() == 2
    im = image[..., None] if single else image
    H, W, C = im.shape
    p = torch.stack([xy[:, 0] * W - 0.5, xy[:, 1] * H - 0.5], dim=1)
    x0 = torch.floor(p[:, 0])
    y0 = torch.floor(p[:, 1])
    fx = (p[:, 0] - x0).unsqueeze(1)
    fy = (p[:, 1] - y0).unsqueeze(1)
    x0l = x0.long().clamp(0, W - 1)
    y0l = y0.long().clamp(0, H - 1)
    x1l = (x0l + 1).clamp(0, W - 1)
    y1l = (y0l + 1).clamp(0, H - 1)
    out = (im[y0l, x0l] * (1 - fx) * (1 - fy)
           + im[y0l, x1l] * fx * (1 - fy)
           + im[y1l, x0l] * (1 - fx) * fy
           + im[y1l, x1l] * fx * fy)
    return out[:, 0] if single else out


def pixel_grid(h: int, w: int, device) -> torch.Tensor:
    """(h*w, 2) pixel centres in [0, 1]^2, row-major."""
    ys = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
    xs = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xv.reshape(-1), yv.reshape(-1)), dim=1)


def warp_image(source: torch.Tensor, u_fn, shape, chunk: int = 1 << 20) -> torch.Tensor:
    """Render target(x) = source(x + u(x)) over the full pixel grid."""
    h, w = shape
    xy = pixel_grid(h, w, source.device)
    px = torch.tensor([w, h], device=source.device, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, xy.shape[0], chunk):
            q = xy[i : i + chunk]
            out.append(sample_bilinear(source, q + u_fn(q) / px))
    return torch.cat(out).reshape(h, w, *source.shape[2:])


# -------------------------------------------------- cross-modal mismatch


def apply_mismatch(image: torch.Tensor, spec: dict, seed: int = 0) -> torch.Tensor:
    """Turn a clean warped image into the second modality's version of it.

    A gamma remap plus an affine intensity change stands in for the different
    contrast transfer of the two microscopes; Poisson-Gaussian noise stands in
    for photon statistics plus camera read noise.  Both break the assumption
    that matching intensities means matching tissue, which is why the loss has
    to become LNCC once this is on.
    """
    out = image.clamp(min=0) ** spec.get("gamma", 1.0)
    out = out * spec.get("intensity_scale", 1.0) + spec.get("intensity_offset", 0.0)
    gain = spec.get("poisson_gain", 0.0)
    if gain:
        g = torch.Generator(device=out.device).manual_seed(seed + 991)
        out = torch.poisson(out.clamp(min=0) * gain, generator=g) / gain
    sigma = spec.get("read_noise_sigma", 0.0)
    if sigma:
        g = torch.Generator(device=out.device).manual_seed(seed + 992)
        out = out + torch.randn(out.shape, device=out.device, generator=g) * sigma
    return out


def gaussian_blur(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur of a (H, W) image; sigma 0 returns it unchanged."""
    if sigma_px <= 0:
        return image
    r = max(1, int(round(3 * sigma_px)))
    k = torch.arange(-r, r + 1, device=image.device, dtype=image.dtype)
    k = torch.exp(-(k**2) / (2 * sigma_px**2))
    k = k / k.sum()
    x = image[None, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[0, 0]


def build_pyramid(image: torch.Tensor, sigmas) -> list:
    """One blurred copy of `image` per sigma, coarsest first."""
    return [gaussian_blur(image, float(s)) for s in sigmas]


def pyramid_level(step: int, steps: int, switch_at) -> int:
    """Which pyramid level a given step is on.

    Registration by intensity is a local search: a patch correlation carries no
    gradient once the misalignment exceeds about half its window, and an L2 term
    none once it exceeds the width of the image feature it sits on.  Blurring
    both images widens those features, so the capture range at the coarsest
    level is set by sigma rather than by the texture.  Without this, a 9 px LNCC
    window cannot recover a 24 px displacement no matter which model is fitting
    it -- the failure is in the objective, not the parameterisation.
    """
    f = step / max(1, steps)
    lvl = 0
    for i, a in enumerate(switch_at):
        if f >= a:
            lvl = i
    return lvl


def patch_offsets(window_px: int, shape, device) -> torch.Tensor:
    """(P*P, 2) offsets of a window_px square patch, in normalised units."""
    h, w = shape
    r = window_px // 2
    d = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    return torch.stack((dx.reshape(-1) / w, dy.reshape(-1) / h), dim=1)


def lncc_loss(pred: torch.Tensor, target: torch.Tensor, n_patches: int) -> torch.Tensor:
    """1 - mean local normalised cross-correlation over (n_patches, P*P) samples."""
    a = pred.reshape(n_patches, -1)
    b = target.reshape(n_patches, -1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    ncc = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)
    return 1.0 - ncc.mean()


# --------------------------------------------------- ground-truth fields


class Fourier:
    """Smooth global warp: a few low-wavenumber modes, scaled to a peak displacement."""

    def __init__(self, n_modes=3, k_max=2, max_displacement_px=24.0, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        k = torch.randint(-k_max, k_max + 1, (4 * n_modes, 2), generator=g)
        k = k[k.abs().sum(1) > 0][:n_modes].float()
        self.k = (2 * math.pi * k).to(device)                        # (M, 2)
        self.phase = (torch.rand(n_modes, 2, generator=g) * 2 * math.pi).to(device)
        self.amp = (torch.randn(n_modes, 2, generator=g)).to(device)
        self.scale = 1.0
        self.scale = float(max_displacement_px / self._raw(_probe(device)).norm(dim=1).max())

    def _raw(self, xy):
        theta = xy @ self.k.T                                        # (N, M)
        return torch.stack([(torch.cos(theta + self.phase[:, d]) * self.amp[:, d]).sum(1)
                            for d in (0, 1)], dim=1)

    def __call__(self, xy):
        return self.scale * self._raw(xy)


class GaussianBumps:
    """Local bending: a handful of compact displacements placed on the foreground."""

    def __init__(self, n_bumps=6, sigma_px=(30.0, 80.0), max_displacement_px=20.0,
                 shape=(1069, 904), centres=None, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed + 101)
        h, w = shape
        if centres is None:
            centres = torch.rand(n_bumps, 2, generator=g)
        self.c = centres[:n_bumps].to(device)                        # (B, 2) in [0,1]^2
        s = torch.rand(n_bumps, generator=g) * (sigma_px[1] - sigma_px[0]) + sigma_px[0]
        # Anisotropic pixels: convert each sigma to normalised units per axis.
        self.sig = torch.stack([s / w, s / h], dim=1).to(device)     # (B, 2)
        d = torch.randn(n_bumps, 2, generator=g)
        self.d = (d / d.norm(dim=1, keepdim=True)).to(device)        # (B, 2) directions
        self.amp = (torch.rand(n_bumps, generator=g) * 0.5 + 0.5).to(device)
        self.scale = 1.0
        self.scale = float(max_displacement_px / self._raw(_probe(device)).norm(dim=1).max())

    def _raw(self, xy):
        r2 = (((xy[:, None, :] - self.c[None]) / self.sig[None]) ** 2).sum(-1)  # (N, B)
        g = torch.exp(-0.5 * r2) * self.amp[None]
        return g @ self.d

    def __call__(self, xy):
        return self.scale * self._raw(xy)


class RigidMotion:
    """A slowly evolving rigid deformation: u(x, t), translation or rotation.

    The point of a time-varying ground truth is that the field is exactly known
    at every instant, so a fit over (x, y, t) can be scored frame by frame --
    including at times between the ones it was trained on. Motion is stated per
    frame, so a longer sequence at the same speed travels further, which is what
    a longer recording does.
    """

    def __init__(self, kind="translation", total=80.0, n_frames=200,
                 shape=(1069, 904), angle_deg=25.0, device="cpu"):
        # TOTAL over the sequence, not per frame. Per frame reads as more
        # physical, but it makes the frame-count sweep change two things at
        # once: at a fixed rate a 100-frame run travels an eighth as far as an
        # 800-frame one, so the comparison is confounded and the short runs are
        # invisible. Fixing the total isolates the temporal sampling, which is
        # what the sweep is for.
        self.kind = kind
        self.total_motion = float(total)
        self.n = int(n_frames)
        self.speed = self.total_motion / max(1, self.n - 1)
        h, w = shape
        self.px = torch.tensor([w, h], device=device, dtype=torch.float32)
        a = math.radians(angle_deg)
        self.dir = torch.tensor([math.cos(a), math.sin(a)], device=device)
        self.centre = torch.tensor([0.5, 0.5], device=device)

    def total(self):
        """Motion accumulated over the whole sequence, for the caption."""
        return self.total_motion

    def __call__(self, xy, t):
        """xy: (N, 2) in [0,1]^2;  t: scalar in [0,1] -> (N, 2) displacement in px."""
        f = float(t) * (self.n - 1)                       # frame index
        if self.kind == "translation":
            return (self.dir * (self.speed * f)).expand(xy.shape[0], 2)
        th = math.radians(self.speed * f)
        # rotation about the centre, in PIXEL space -- doing it in normalised
        # coordinates would shear the result on a non-square image
        d = (xy - self.centre) * self.px
        c, s = math.cos(th), math.sin(th)
        rot = torch.stack([d[:, 0] * c - d[:, 1] * s,
                           d[:, 0] * s + d[:, 1] * c], dim=1)
        return rot - d

    def at(self, t):
        """A plain callable for one instant, for warp_image()."""
        return lambda xy: self(xy, t)


class MovingBand:
    """A slip band that travels or turns: the deformation PATTERN moves, not the image.

    The distinction matters. Rigidly translating the picture gives a field that
    is the same constant everywhere and merely grows with t -- an encoder can
    fit that with almost no spatial capacity at all. Here the field has a sharp
    feature at a definite place, and that place changes with time, so the
    encoder has to put detail somewhere different in every frame. That is the
    shape of a real moving deformation, and the one this repo's spatial results
    were about.

    u(x, t) = offset * clamp(s / width, -1, 1) * parallel,  s = (x - c(t)) . normal

    `kind="translate"` sweeps the band's centre along its own normal;
    `kind="rotate"` turns it about the middle of the image. `total` is the
    distance in pixels or the angle in degrees covered across the whole
    sequence, so a frame-count sweep changes the sampling and not the motion.
    """

    def __init__(self, kind="translate", total=300.0, n_frames=200,
                 shape=(1069, 904), width_px=12.0, offset_px=18.0,
                 angle_deg=35.0, device="cpu"):
        self.kind = kind
        self.total_motion = float(total)
        self.n = int(n_frames)
        self.speed = self.total_motion / max(1, self.n - 1)
        h, w = shape
        self.px = torch.tensor([w, h], device=device, dtype=torch.float32)
        self.centre = torch.tensor([0.5, 0.5], device=device)
        self.width = float(width_px)
        self.offset = float(offset_px)
        self.angle0 = float(angle_deg)
        self.device = device

    def total(self):
        return self.total_motion

    def __call__(self, xy, t):
        """xy: (N, 2) in [0,1]^2;  t: a scalar or an (N, 1) tensor in [0, 1]."""
        tt = (t if torch.is_tensor(t) else
              torch.full((xy.shape[0], 1), float(t), device=xy.device))
        if tt.dim() == 1:
            tt = tt.unsqueeze(1)
        deg = self.angle0 + (self.total_motion * tt if self.kind == "rotate" else 0.0)
        th = torch.deg2rad(deg if torch.is_tensor(deg)
                           else torch.full_like(tt, float(deg)))
        c, s = torch.cos(th), torch.sin(th)
        nrm = torch.cat([-s, c], dim=1)                        # (N, 2)
        par = torch.cat([c, s], dim=1)
        # the band centre travels along its own normal, centred on the middle
        shift = (self.total_motion * (tt - 0.5)) if self.kind == "translate" else \
                torch.zeros_like(tt)
        d = (xy - self.centre) * self.px - nrm * shift
        sdist = (d * nrm).sum(1, keepdim=True)                 # signed distance, px
        prof = (sdist / self.width).clamp(-1.0, 1.0)
        return self.offset * prof * par

    def at(self, t):
        return lambda xy: self(xy, t)


class MultiScaleBands:
    """Displacement whose spatial SCALE varies across the frame, at constant amplitude.

    Every other ground truth here has one characteristic scale everywhere, so a
    level decomposition of the fit has nothing to report: whichever level matches
    that scale carries the whole field. This one is built in vertical bands whose
    Gaussian width falls geometrically from left to right while their peak
    displacement stays the same, so the *frequency* varies and the amplitude does
    not. A hash grid that allocates by scale should then use visibly coarser cells
    on the left than on the right -- and if it does not, the claim that it puts
    capacity where the data needs it is not doing any work here.
    """

    def __init__(self, n_bands=4, sigma_px=(128.0, 16.0), max_displacement_px=8.0,
                 per_band=8, shape=(1069, 904), seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed + 303)
        h, w = shape
        cs, sg, dr, bid = [], [], [], []
        self.n_bands = n_bands
        for b in range(n_bands):
            f = b / max(1, n_bands - 1)
            sig = sigma_px[0] * (sigma_px[1] / sigma_px[0]) ** f     # geometric: one step finer per band
            # MORE BUMPS WHERE THEY ARE SMALLER. With a fixed count per band, a
            # narrow bump covers a fraction of the band's area and the mean
            # displacement there falls with sigma -- 15x across four bands when
            # measured. The level decomposition weights by how much a level moves
            # the field, so the fine bands would barely register and the panel
            # would be reporting amplitude again instead of scale.
            n_b = int(round(per_band * sigma_px[0] / sig))
            x0, x1 = b / n_bands, (b + 1) / n_bands
            cx = x0 + (x1 - x0) * torch.rand(n_b, generator=g)
            cy = torch.rand(n_b, generator=g)
            cs.append(torch.stack([cx, cy], 1))
            sg.append(torch.stack([torch.full((n_b,), sig / w),
                                   torch.full((n_b,), sig / h)], 1))
            d = torch.randn(n_b, 2, generator=g)
            dr.append(d / d.norm(dim=1, keepdim=True))
            bid.append(torch.full((n_b,), b))
        self.c = torch.cat(cs).to(device)
        self.sig = torch.cat(sg).to(device)
        self.d = torch.cat(dr).to(device)
        self.band = torch.cat(bid).to(device)
        self.amp = torch.ones(self.c.shape[0], device=device)
        self.scale = 1.0
        # EQUAL PEAK PER BAND, so the only thing that varies across the frame is
        # the scale. Measured per band on a probe grid rather than assumed,
        # because overlapping wide bumps reach a higher peak than isolated
        # narrow ones at the same per-bump amplitude.
        probe = _probe(device)
        px = torch.tensor([w, h], device=device, dtype=torch.float32)
        for b in range(n_bands):
            keep = (self.band == b).float()
            u = (torch.exp(-0.5 * (((probe[:, None, :] - self.c[None]) / self.sig[None]) ** 2
                                   ).sum(-1)) * (self.amp * keep)[None]) @ self.d
            peak = float(u.norm(dim=1).max())
            if peak > 0:
                self.amp = torch.where(self.band == b, self.amp / peak, self.amp)
        self.scale = float(max_displacement_px
                           / self._raw(probe).norm(dim=1).max())
    def _raw(self, xy):
        r2 = (((xy[:, None, :] - self.c[None]) / self.sig[None]) ** 2).sum(-1)
        return (torch.exp(-0.5 * r2) * self.amp[None]) @ self.d

    def __call__(self, xy):
        return self.scale * self._raw(xy)


class ShearBand:
    """A near-discontinuity: displacement parallel to a band, flipping across it."""

    def __init__(self, angle_deg=35.0, width_px=12.0, offset_px=18.0,
                 shape=(1069, 904), centre=(0.5, 0.5), device="cpu"):
        a = math.radians(angle_deg)
        h, w = shape
        self.par = torch.tensor([math.cos(a), math.sin(a)], device=device)
        self.nrm = torch.tensor([-math.sin(a), math.cos(a)], device=device)
        self.centre = torch.tensor(centre, device=device)
        self.px = torch.tensor([w, h], device=device, dtype=torch.float32)
        self.width = width_px
        self.offset = offset_px

    def __call__(self, xy):
        s = ((xy - self.centre) * self.px) @ self.nrm                # signed distance, px
        prof = (s / self.width).clamp(-1.0, 1.0)                     # sharp but continuous
        return self.offset * prof.unsqueeze(1) * self.par


class Sum:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, xy):
        out = self.terms[0](xy)
        for t in self.terms[1:]:
            out = out + t(xy)
        return out


def _probe(device, n=256):
    return pixel_grid(n, n, device)


def build_deformation(spec: dict, built: dict, shape, device, seed=0, foreground=None):
    """Instantiate one entry of the `deformations:` list."""
    kind = spec["type"]
    if kind == "fourier":
        return Fourier(spec.get("n_modes", 3), spec.get("k_max", 2),
                       spec.get("max_displacement_px", 24.0), seed, device)
    if kind == "gaussian_bumps":
        centres = None
        if spec.get("placement") == "foreground" and foreground is not None:
            idx = torch.nonzero(foreground.reshape(-1), as_tuple=False).squeeze(1)
            g = torch.Generator(device=idx.device).manual_seed(seed + 7)
            pick = idx[torch.randint(len(idx), (spec.get("n_bumps", 6),),
                                     generator=g, device=idx.device)]
            h, w = shape
            centres = torch.stack([(pick % w).float() / w, (pick // w).float() / h], dim=1)
        return GaussianBumps(spec.get("n_bumps", 6), tuple(spec.get("sigma_px", (30, 80))),
                             spec.get("max_displacement_px", 20.0), shape, centres,
                             seed, device)
    if kind == "multiscale_bands":
        return MultiScaleBands(spec.get("n_bands", 4),
                               tuple(spec.get("sigma_px", (128.0, 16.0))),
                               spec.get("max_displacement_px", 8.0),
                               spec.get("per_band", 8), shape, seed, device)
    if kind == "shear_band":
        centre = (0.5, 0.5)
        if spec.get("placement") == "foreground" and foreground is not None:
            idx = torch.nonzero(foreground.reshape(-1), as_tuple=False).squeeze(1)
            h, w = shape
            p = idx[len(idx) // 2]
            centre = (float(p % w) / w, float(p // w) / h)
        return ShearBand(spec.get("angle_deg", 35.0), spec.get("width_px", 12.0),
                         spec.get("offset_px", 18.0), shape, centre, device)
    if kind == "sum":
        return Sum([built[t] for t in spec["terms"]])
    raise ValueError(f"unknown deformation type {kind!r}")


# ------------------------------------------------------------- the models


class ControlGrid(nn.Module):
    """A dense grid of displacement control points, bilinearly interpolated.

    The classical registration parameterisation: (gh, gw, 2) free parameters and
    nothing else.  Smooth by construction below the control spacing, which is
    both its strength (the unconstrained background stays sane) and its ceiling
    (it cannot bend faster than one control cell).
    """

    def __init__(self, grid=(16, 16), output_scale_px=40.0):
        super().__init__()
        gh, gw = grid
        self.u = nn.Parameter(torch.zeros(gh, gw, 2))
        self.scale = output_scale_px
        self.grid = (gh, gw)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # Hand-rolled bilinear rather than F.grid_sample: grid_sample has no
        # double backward, and the folding penalty differentiates the Jacobian.
        return sample_bilinear(self.u, xy) * self.scale

    def n_parameters(self):
        return self.u.numel(), 0


class SirenDeform(nn.Module):
    """A SIREN producing a pixel displacement, with a band window.

    The twin of ngp-demo's HashGridDeform: same signature, same output_scale_px, and
    `set_level_window` kept under that name so the trainer's coarse-to-fine schedule
    does not have to know which representation it is driving.  Here it releases the
    first layer's units low frequency first instead of releasing grid levels.
    """

    def __init__(self, siren: dict, output_scale_px=40.0, n_bands=16):
        super().__init__()
        self.field = SirenField(n_input_dims=2, n_output_dims=2,
                                output_activation="none", **siren)
        self.scale = output_scale_px
        self.n_bands = n_bands

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(xy) * self.scale

    def set_level_window(self, alpha):
        self.field.set_band_window(alpha, self.n_bands)

    def n_parameters(self):
        return self.field.n_parameters()


def build_model(spec: dict, device):
    if spec["kind"] == "control_grid":
        m = ControlGrid(tuple(spec.get("grid", (16, 16))),
                        spec.get("output_scale_px", 40.0))
    elif spec["kind"] == "siren":
        m = SirenDeform(dict(spec["siren"]), spec.get("output_scale_px", 40.0),
                        int(spec.get("n_bands", 16)))
    else:
        raise ValueError(f"unknown model kind {spec['kind']!r}")
    return m.to(device)


# ------------------------------------------------------------- derivatives


def field_jacobian(u_fn, xy: torch.Tensor, px: torch.Tensor, create_graph=False):
    """d(x + u)/dx in pixel units. xy: (N, 2) -> (N, 2, 2).

    Rows are the two output components, columns the two input axes, so
    det(J) < 0 marks a fold.
    """
    x = xy.detach().requires_grad_(True)
    u = u_fn(x)
    rows = []
    for d in (0, 1):
        (g,) = torch.autograd.grad(u[:, d].sum(), x, create_graph=True)
        rows.append(g / px)                       # d u_d / d x_j, both in pixels
    J = torch.stack(rows, dim=1)
    J = J + torch.eye(2, device=xy.device)
    return J if create_graph else J.detach()
