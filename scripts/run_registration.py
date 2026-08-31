#!/usr/bin/env python
"""Registration benchmark: recover a known deformation with an NGP or a control grid.

    python scripts/run_registration.py --config config/registration_benchmark.yaml
    python scripts/run_registration.py --deformations local_bending --models ngp tensor_16

`target(x) = source(x + u_gt(x))` with `u_gt` analytic, so every run is scored on
the *field* it recovered -- split into the textured foreground, the black
background where no data constrains it, and the band between -- and not only on
how well the warped image matches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren.deform import (apply_mismatch, build_deformation, build_model, build_pyramid,
                        field_jacobian, lncc_loss, patch_offsets, pixel_grid,
                        pyramid_level, sample_bilinear, warp_image)
from siren.utils import psnr, read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config/registration_benchmark.yaml"))
    p.add_argument("--deformations", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--mismatch", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=None, help="override config")
    p.add_argument("--no-pyramid", action="store_true",
                   help="disable the image blur schedule, to test what the "
                        "encoder's own level window is worth on its own")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve_inherits(entries):
    """Expand `inherits: <name>` in a list of dicts, child keys winning."""
    by_name = {e["name"]: e for e in entries}
    out = []
    for e in entries:
        merged = dict(e)
        while "inherits" in merged:
            parent = dict(by_name[merged.pop("inherits")])
            parent.pop("name", None)
            parent.pop("inherits", None)
            for k, v in parent.items():
                merged.setdefault(k, v)
        out.append(merged)
    return out


def load_image(cfg, device):
    img = read_image(os.path.join(ROOT, cfg["path"]))
    t = torch.from_numpy(img).to(device)
    if cfg.get("channels", "gray") == "gray":
        t = t.mean(-1)
    d = int(cfg.get("downsample", 1))
    if d > 1:
        t = F.avg_pool2d(t[None, None], d)[0, 0] if t.dim() == 2 else \
            F.avg_pool2d(t.permute(2, 0, 1)[None], d)[0].permute(1, 2, 0)
    return t


def foreground_mask(image, cfg):
    """Blur, threshold, dilate -- the lit part of the canvas."""
    im = image if image.dim() == 2 else image.mean(-1)
    k = max(1, int(cfg.get("blur_px", 3))) | 1
    g = torch.arange(k, device=im.device).float() - k // 2
    g = torch.exp(-(g**2) / (2 * (k / 3) ** 2))
    g = g / g.sum()
    b = F.conv2d(im[None, None], g.view(1, 1, 1, -1), padding=(0, k // 2))
    b = F.conv2d(b, g.view(1, 1, -1, 1), padding=(k // 2, 0))[0, 0]
    m = b > cfg.get("threshold", 0.06)
    # Opening first: the dark surround has craquelure highlights above threshold,
    # and left alone they survive dilation as isolated islands that would act as
    # free landmarks in a region meant to carry no data.
    o = int(cfg.get("open_px", 0))
    if o > 0:
        m = -F.max_pool2d(-m.float()[None, None], 2 * o + 1, stride=1, padding=o)
        m = F.max_pool2d(m, 2 * o + 1, stride=1, padding=o)[0, 0] > 0.5
    r = int(cfg.get("dilate_px", 12))
    if r > 0:
        m = F.max_pool2d(m.float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0.5
    return m


def _feather(mask, r):
    """Soft alpha from a binary mask: a box blur of radius r, in [0, 1]."""
    if r <= 0:
        return mask.float()
    k = 2 * r + 1
    a = F.avg_pool2d(mask.float()[None, None], k, stride=1, padding=r)[0, 0]
    return a.clamp(0, 1)


def sample_points(fg_idx, n, frac, shape, device):
    """`frac` of the batch from foreground pixels (with sub-pixel jitter), rest uniform."""
    h, w = shape
    n_fg = int(n * frac)
    idx = fg_idx[torch.randint(len(fg_idx), (n_fg,), device=device)]
    xy_fg = torch.stack([(idx % w).float(), (idx // w).float()], dim=1)
    xy_fg = (xy_fg + torch.rand(n_fg, 2, device=device)) / torch.tensor([w, h], device=device)
    xy_u = torch.rand(n - n_fg, 2, device=device)
    return torch.cat([xy_fg, xy_u], dim=0)


def _weight(w, loss_kind):
    return float(w[loss_kind]) if isinstance(w, dict) else float(w)


def train_one(model, source, target, fg_idx, shape, cfg, device, loss_kind="l2",
              log=print):
    tr = cfg["training"]
    px = torch.tensor([shape[1], shape[0]], device=device, dtype=torch.float32)
    kind = "siren" if hasattr(model, "set_level_window") else "control_grid"
    lr = tr["lr"][kind]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = tr["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * tr.get("lr_final_fraction", 0.03))
    c2f = cfg["_model"].get("coarse_to_fine", {})
    reg_n = tr.get("reg_batch", 8192)
    offs = patch_offsets(tr["loss"]["lncc"]["window_px"], shape, device)
    n_p = tr["loss"]["lncc"]["n_patches"]
    # Weights are per loss kind: the L2 data term sits near 1e-4 while 1-LNCC
    # sits near 3e-1, so a single absolute weight makes the regulariser ~1000x
    # weaker in the cross-modal arm and folding gets blamed on the mismatch.
    w_smooth = _weight(tr["loss"]["smoothness"]["weight"], loss_kind)
    w_fold = _weight(tr["loss"]["folding"]["weight"], loss_kind)

    pyr = tr.get("image_pyramid", {"sigma_px": [0], "switch_at": [0.0]})
    src_p = build_pyramid(source, pyr["sigma_px"])
    tgt_p = build_pyramid(target, pyr["sigma_px"])

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    hist, t_train = [], 0.0
    lvl = -1
    for step in range(steps + 1):
        t0 = time.perf_counter()
        if kind == "siren" and c2f.get("enabled"):
            n_lv = getattr(model, "n_bands", 16)
            a = c2f["start_levels"] + (n_lv - c2f["start_levels"]) * min(
                1.0, step / max(1, c2f["full_at_step"]))
            model.set_level_window(a)

        if loss_kind == "lncc":
            n_p = tr["loss"]["lncc"]["n_patches"]
            c = sample_points(fg_idx, n_p, tr["sampling"]["foreground_fraction"],
                              shape, device)
            xy = (c[:, None, :] + offs[None]).reshape(-1, 2)
        else:
            xy = sample_points(fg_idx, tr["batch"], tr["sampling"]["foreground_fraction"],
                               shape, device)
        new_lvl = pyramid_level(step, steps, pyr["switch_at"])
        if new_lvl != lvl:
            lvl = new_lvl
            log(f"    pyramid sigma {pyr['sigma_px'][lvl]} px at step {step}")
        u = model(xy)
        pred = sample_bilinear(src_p[lvl], xy + u / px)
        gt = sample_bilinear(tgt_p[lvl], xy)
        loss = (lncc_loss(pred, gt, n_p) if loss_kind == "lncc"
                else ((pred - gt) ** 2).mean())
        photo = loss.item()

        # Regularisers on a subsample: both need d u / d x, which costs two extra
        # backward passes through the model.
        if w_smooth > 0 or w_fold > 0:
            xr = sample_points(fg_idx, reg_n, tr["sampling"]["foreground_fraction"],
                               shape, device)
            J = field_jacobian(model, xr, px, create_graph=True)
            dudx = J - torch.eye(2, device=device)
            if w_smooth > 0:
                loss = loss + w_smooth * (dudx**2).sum((1, 2)).mean()
            if w_fold > 0:
                det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                loss = loss + w_fold * F.relu(0.1 - det).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        if step % max(1, steps // 10) == 0 or step == steps:
            hist.append({"step": step, "train_s": t_train, "loss": photo})
            log(f"    step {step:5d}  {t_train:6.1f}s  loss {photo:.3e}")
    peak = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0
    return hist, t_train, peak


@torch.no_grad()
def _dense_field(u_fn, shape, device, chunk=1 << 18):
    xy = pixel_grid(*shape, device)
    return torch.cat([u_fn(xy[i:i + chunk]) for i in range(0, len(xy), chunk)])


def evaluate(model, u_gt, source, target_clean, target_obs, masks, shape, device):
    h, w = shape
    px = torch.tensor([w, h], device=device, dtype=torch.float32)
    warped = warp_image(source, model, shape)
    m = {}
    # PSNR against the *clean* warp, which stays meaningful when the observed
    # target has been gamma-remapped and noised; NCC against what was optimised.
    m["psnr_warped_vs_clean_target"] = psnr(warped, target_clean)
    a = warped.reshape(-1) - warped.mean()
    b = target_obs.reshape(-1) - target_obs.mean()
    m["ncc_warped_vs_observed_target"] = float((a * b).sum() / (a.norm() * b.norm()))

    u = _dense_field(model, shape, device)
    ugt = _dense_field(u_gt, shape, device)
    epe = (u - ugt).norm(dim=1)
    m["endpoint_error_px_mean"] = float(epe.mean())
    m["endpoint_error_px_p95"] = float(torch.quantile(epe[::7].float(), 0.95))
    for name, mask in masks.items():
        m[f"endpoint_error_px_{name}"] = float(epe[mask.reshape(-1)].mean())

    sub = pixel_grid(h // 3, w // 3, device)
    J = field_jacobian(model, sub, px)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    m["jacobian_det_min"] = float(det.min())
    m["jacobian_det_negative_fraction"] = float((det < 0).float().mean())

    # Bending energy from second differences of the rendered field: comparable
    # across models, and unlike an autograd Laplacian it does not vanish for a
    # bilinear parameterisation.
    f = u.reshape(h, w, 2)
    d2 = (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1])
    m["bending_energy"] = float((d2**2).sum(-1).mean())
    return m, warped.cpu(), epe.reshape(h, w).cpu(), u.reshape(h, w, 2).cpu()


def run_figure(out, tag, source, target, warped, epe, u, ugt, shape, cfg):
    h, w = shape
    fig, ax = plt.subplots(1, 5, figsize=(22, 5.2))
    panels = [(source.cpu(), "gray", (0, 1), "source"),
              (target.cpu(), "gray", (0, 1), "target"),
              (warped, "gray", (0, 1), "source warped by the fit"),
              (epe, "inferno", (0, float(np.percentile(epe.numpy(), 99))),
               "endpoint error (px)")]
    for a, (img, cmap, lim, note) in zip(ax, panels):
        im = a.imshow(img.numpy(), cmap=cmap, vmin=lim[0], vmax=lim[1])
        a.set_xticks([]); a.set_yticks([])
        a.text(0.98, 0.02, note, transform=a.transAxes, va="bottom", ha="right",
               fontsize=11, color="w")
        if cmap != "gray":
            fig.colorbar(im, ax=a, fraction=0.046)

    a = ax[4]
    sp = cfg["visualization"]["warped_grid_overlay"]["spacing_px"]
    a.imshow(target.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    _draw_warped_grid(a, ugt.reshape(h, w, 2), sp, "tab:red", lw=1.2)
    _draw_warped_grid(a, u, sp, "tab:blue", lw=0.8, dashes=(4, 3))
    a.set_xlim(0, w); a.set_ylim(h, 0)
    a.set_xticks([]); a.set_yticks([])
    a.text(0.98, 0.02, "grid: ground truth (red) vs fit (blue)", transform=a.transAxes,
           va="bottom", ha="right", fontsize=11, color="w")
    for i, a in enumerate(ax):
        a.text(0.02, 0.98, "abcde"[i], transform=a.transAxes, va="top", ha="left",
               fontsize=16, fontweight="bold", color="w")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}.png"), dpi=cfg["visualization"].get("dpi", 130))
    plt.close(fig)


def _draw_warped_grid(ax, u, spacing, color, lw=0.6, dashes=None):
    """Draw the image of a regular grid under x -> x + u(x)."""
    h, w = u.shape[:2]
    ys = np.arange(0, h, spacing)
    xs = np.arange(0, w, spacing)
    un = u.numpy()
    kw = {"color": color, "lw": lw, "alpha": 0.9}
    if dashes:
        kw["dashes"] = dashes
    for y in ys:
        ax.plot(xs + un[y, xs, 0], y + un[y, xs, 1], **kw)
    for x in xs:
        ax.plot(x + un[ys, x, 0], ys + un[ys, x, 1], **kw)


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps:
        cfg["training"]["steps"] = args.steps
    if args.no_pyramid:
        cfg["training"]["image_pyramid"] = {"sigma_px": [0], "switch_at": [0.0]}
    out_root = args.out or os.path.join(ROOT, cfg["runs"]["out"])
    os.makedirs(out_root, exist_ok=True)
    device = torch.device(args.device or cfg.get("device", "cuda"))
    torch.manual_seed(cfg.get("seed", 0))

    source = load_image(cfg["image"], device)
    shape = tuple(source.shape[:2])
    fgc = cfg["image"]["foreground"]
    fg = foreground_mask(source, fgc)
    if fgc.get("zero_background"):
        # The painting's surround is dark but textured; the zebrafish surround is
        # empty. Zeroing it (through a feathered edge, so the boundary is not
        # itself a registration cue) makes the background genuinely data-free,
        # which is the regime the benchmark is meant to probe.
        source = source * _feather(fg, fgc.get("feather_px", 9))
    fg_idx = torch.nonzero(fg.reshape(-1), as_tuple=False).squeeze(1)
    band_r = cfg["metrics"]["evaluation"]["boundary_band_px"]
    grown = F.max_pool2d(fg.float()[None, None], 2 * band_r + 1, stride=1,
                         padding=band_r)[0, 0] > 0.5
    masks = {"foreground": fg, "background": ~grown, "boundary_band": grown & ~fg}
    print(f"image {shape[1]}x{shape[0]}   foreground {float(fg.float().mean())*100:.1f}% "
          f"of pixels, background {float(masks['background'].float().mean())*100:.1f}%")

    deform_specs = {d["name"]: d for d in cfg["deformations"]}
    models = {m["name"]: m for m in resolve_inherits(cfg["models"])}
    want_d = args.deformations or list(deform_specs)
    want_m = args.models or list(models)

    built = {}
    for name, spec in deform_specs.items():
        built[name] = build_deformation(spec, built, shape, device,
                                        cfg.get("seed", 0), fg)

    mismatches = {m["name"]: m for m in cfg.get("modality_mismatch",
                                                [{"name": "matched", "loss": "l2"}])}
    want_x = args.mismatch or list(mismatches)

    rows = []
    for dname in want_d:
        u_gt = built[dname]
        target_clean = warp_image(source, u_gt, shape)
        ugt_dense = _dense_field(u_gt, shape, device).reshape(*shape, 2).cpu()
        mag = ugt_dense.norm(dim=-1)
        print(f"\n{dname}: |u_gt| mean {mag.mean():.2f} px, max {mag.max():.2f} px")
        for xname in want_x:
            xspec = mismatches[xname]
            target_obs = apply_mismatch(target_clean, xspec, cfg.get("seed", 0))
            loss_kind = xspec.get("loss", "l2")
            print(f"  mismatch {xname} (loss {loss_kind})")
            for mname in want_m:
                spec = models[mname]
                print(f"    {mname}")
                torch.manual_seed(cfg.get("seed", 0))
                model = build_model(spec, device)
                cfg["_model"] = spec
                hist, secs, peak = train_one(model, source, target_obs, fg_idx, shape,
                                             cfg, device, loss_kind,
                                             log=lambda s: print("  " + s))
                m, warped, epe, u = evaluate(model, u_gt, source, target_clean,
                                             target_obs, masks, shape, device)
                n_a, n_b = model.n_parameters()
                pyr_sig = cfg["training"].get("image_pyramid", {}).get("sigma_px", [0])
                m.update({"deformation": dname, "mismatch": xname, "model": mname,
                          "loss": loss_kind,
                          "pyramid": bool(max(pyr_sig) > 0),
                          "w_smooth": _weight(cfg["training"]["loss"]["smoothness"]["weight"],
                                              loss_kind),
                          "w_fold": _weight(cfg["training"]["loss"]["folding"]["weight"],
                                            loss_kind), "n_parameters": n_a + n_b,
                          "train_seconds": secs, "peak_gpu_mb": peak})
                rows.append(m)
                tag = f"{dname}__{xname}__{mname}"
                torch.save({"state_dict": model.state_dict(), "model_spec": spec,
                            "deformation": dname, "mismatch": xname},
                           os.path.join(out_root, f"{tag}.pt"))
                run_figure(out_root, tag, source, target_obs, warped, epe, u,
                           ugt_dense, shape, cfg)
                print(f"      PSNR {m['psnr_warped_vs_clean_target']:.2f} dB   EPE "
                      f"fg {m['endpoint_error_px_foreground']:.3f} / band "
                      f"{m['endpoint_error_px_boundary_band']:.3f} / bg "
                      f"{m['endpoint_error_px_background']:.3f} px   "
                      f"det<0 {m['jacobian_det_negative_fraction']*100:.2f}%")

    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    write_table(rows, out_root)
    print(f"\nwrote {out_root}")


def write_table(rows, out):
    cols = [("model", "model", "{}"),
            ("psnr_warped_vs_clean_target", "PSNR", "{:.2f}"),
            ("ncc_warped_vs_observed_target", "NCC", "{:.4f}"),
            ("endpoint_error_px_foreground", "EPE fg", "{:.3f}"),
            ("endpoint_error_px_boundary_band", "EPE band", "{:.3f}"),
            ("endpoint_error_px_background", "EPE bg", "{:.3f}"),
            ("jacobian_det_min", "min|J|", "{:.3f}"),
            ("jacobian_det_negative_fraction", "folded", "{:.4f}"),
            ("bending_energy", "bending", "{:.2e}"),
            ("n_parameters", "params", "{:,}"),
            ("train_seconds", "s", "{:.0f}")]
    lines = []
    for key in dict.fromkeys((r["deformation"], r["mismatch"]) for r in rows):
        lines += [f"\n### {key[0]} / {key[1]}\n",
                  "| " + " | ".join(c[1] for c in cols) + " |",
                  "|" + "---|" * len(cols)]
        for r in [r for r in rows
                  if (r["deformation"], r["mismatch"]) == key]:
            lines.append("| " + " | ".join(c[2].format(r[c[0]]) for c in cols) + " |")
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
