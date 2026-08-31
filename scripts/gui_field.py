#!/usr/bin/env python
"""Browser GUI for the registration benchmark: fit a known warp, watch it converge.

    python scripts/gui_field.py                 # http://localhost:8021
    python scripts/gui_field.py --port 8030

Everything the benchmark decides from a YAML file is a control here instead:
which ground-truth deformation to recover, how mismatched the two "modalities"
are, which parameterisation fits it, and -- the part a config file hides -- the
training schedule itself. Learning rate, iteration count, batch size and the two
regulariser weights are sliders, because on this problem they change the answer
as much as the model does.

The panels are chosen so that the two things that separate the methods are both
on screen at once: the warped image (which almost always looks fine) and the
endpoint error against the analytic field (which does not).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren.deform import (apply_mismatch, build_deformation, build_model, build_pyramid,
                        field_jacobian, lncc_loss, patch_offsets, pixel_grid,
                        pyramid_level, sample_bilinear, warp_image)
from siren.utils import psnr
from siren.webui import ABOUT_HTML, CSS, INTERFACE_REG, flow_png
from scripts.run_registration import (_dense_field, _feather, foreground_mask, load_image,
                                      resolve_inherits, sample_points)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixed, not per-refresh: an auto-scaled error panel brightens as the fit
# improves, because the colour tracks that frame's own 99th percentile. 10 px is
# above the band error of every configuration and below the background, where
# the ground-truth warp is simply unreachable.
EPE_LUT_MAX = 10.0
# The photometric residual on its own fixed scale, in intensity units. Against
# the CLEAN warp, not the observed one: under a gamma remap the observed target
# differs in intensity everywhere, and a residual against it would show the
# remap rather than the misalignment.
RESID_LUT_MAX = 0.10
LEVEL_LUT_MAX = 20
DISPLAY_H = 460                      # panel height in px; images are sent downsampled

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "grid": {}, "note": "", "stamp": 0,
       "pyramid_sigma": 0, "levels": None, "switches": [], "levels_live": None}
LOCK = threading.Lock()
STOP = threading.Event()
SCENE = {}                           # cached image / masks / ground truth per key


# --------------------------------------------------------------- rendering


def _png(rgb: np.ndarray) -> str:
    im = Image.fromarray(rgb)
    if im.height > DISPLAY_H:
        im = im.resize((max(1, round(im.width * DISPLAY_H / im.height)), DISPLAY_H),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_png(t: torch.Tensor) -> str:
    a = np.clip(t.detach().cpu().numpy(), 0, 1)
    return _png((np.stack([a] * 3, -1) * 255).astype(np.uint8))


def cmap_png(a: np.ndarray, vmax: float, name="inferno") -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    return _png((matplotlib.colormaps[name](x)[..., :3] * 255).astype(np.uint8))


def grid_lines(u: np.ndarray, spacing: int, shape):
    """Polylines of a regular grid carried through x -> x + u(x), in image px."""
    h, w = shape
    ys = np.arange(0, h, spacing)
    xs = np.arange(0, w, spacing)
    out = []
    for y in ys:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for x in xs])
    for x in xs:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for y in ys])
    return out


# ------------------------------------------------------------------ scene


def get_scene(cfg, dname, xname, device):
    key = (dname, xname)
    if key in SCENE:
        return SCENE[key]
    src = load_image(cfg["image"], device)
    shape = tuple(src.shape[:2])
    fgc = cfg["image"]["foreground"]
    fg = foreground_mask(src, fgc)
    if fgc.get("zero_background"):
        src = src * _feather(fg, fgc.get("feather_px", 9))
    band_r = cfg["metrics"]["evaluation"]["boundary_band_px"]
    grown = F.max_pool2d(fg.float()[None, None], 2 * band_r + 1, stride=1,
                         padding=band_r)[0, 0] > 0.5
    built = {}
    for name, spec in {d["name"]: d for d in cfg["deformations"]}.items():
        built[name] = build_deformation(spec, built, shape, device, cfg.get("seed", 0), fg)
    u_gt = built[dname]
    clean = warp_image(src, u_gt, shape)
    xspec = {m["name"]: m for m in cfg["modality_mismatch"]}[xname]
    obs = apply_mismatch(clean, xspec, cfg.get("seed", 0))
    ugt = _dense_field(u_gt, shape, device).reshape(*shape, 2)
    SCENE[key] = {
        "source": src, "clean": clean, "observed": obs, "u_gt": u_gt,
        "ugt_dense": ugt, "shape": shape, "loss": xspec.get("loss", "l2"),
        "fg_idx": torch.nonzero(fg.reshape(-1), as_tuple=False).squeeze(1),
        "masks": {"foreground": fg, "background": ~grown, "boundary_band": grown & ~fg},
    }
    return SCENE[key]


# --------------------------------------------------------------- training


@torch.no_grad()
def level_maps(model, shape, device, px, block_px=64, sub=4, thresh=0.08):
    """Which frequency band carries the displacement, per block.

    Same decomposition the image page draws, applied to the deformation instead
    of the picture: render u with bands 0..k for every k, difference consecutive
    renders, and the band whose release changes u most in a block is the one
    doing the work there. Control grids have no bands, so this returns nothing
    for them and the panel says so.
    """
    fld = getattr(model, "field", None)
    if fld is None or not hasattr(fld, "set_band_window"):
        return None
    n_bands = getattr(model, "n_bands", 16)
    h, w = shape
    hs, ws = max(8, h // sub), max(8, w // sub)
    xy = pixel_grid(hs, ws, device)
    bs = max(2, block_px // sub)
    prev, deltas = None, []
    lad = ladder_of(fld, w, n_bands)
    for k in range(n_bands + 1):
        fld.set_band_window(float(k), n_bands)
        out = model(xy).reshape(hs, ws, 2)
        if prev is not None:
            deltas.append((out - prev).norm(dim=-1))
        prev = out
    fld.set_band_window(float(n_bands), n_bands)
    D = torch.stack(deltas)
    # WHAT EACH BAND CONTRIBUTES, AND HOW EVENLY -- not "which band works in
    # this block", which is the question ngp-demo asks of a grid and which has no
    # answer here.  A grid level only touches its own cells, so a block can name
    # one.  A SIREN band is a plane wave across the whole frame: measured on the
    # slip_band fit, the busiest tenth of the blocks holds 18-33% of any band's
    # total displacement (10% is perfectly uniform) and every band's amplitude is
    # within a factor of four of every other's, so the block map returned band 15
    # in 239 of 255 blocks.  True, and useless.
    nb = F.avg_pool2d(D[None], bs, stride=bs, ceil_mode=True)[0]      # (B, Hb, Wb)
    flat = nb.reshape(nb.shape[0], -1)
    k = max(1, int(0.1 * flat.shape[1]))
    top = torch.topk(flat, k, dim=1).values.sum(1) / flat.sum(1).clamp(min=1e-12)
    lad = ladder_of(fld, w, n_bands)
    profile = [{"band": b, "mean": float(flat[b].mean()), "top10": float(top[b]),
                "px": lad.get(b, float(w))} for b in range(flat.shape[0])]
    return {"profile": profile, "w": w, "h": h, "n_levels": n_bands}


def ladder_of(field, w, n_bands):
    """band -> px per cycle, the wavelength the overlay draws."""
    f = field.frequencies().cpu()
    band = field.band_of(n_bands).cpu()
    out = {}
    for b in range(n_bands):
        fb = f[band == b]
        if len(fb):
            out[b] = round(w / max(float(fb.mean()), 1e-6), 2)
    return out


def _evaluate(model, sc, device, spacing):
    shape = sc["shape"]
    h, w = shape
    px = torch.tensor([w, h], device=device, dtype=torch.float32)
    warped = warp_image(sc["source"], model, shape)
    u = _dense_field(model, shape, device).reshape(h, w, 2)
    epe = (u - sc["ugt_dense"]).norm(dim=-1)
    sub = pixel_grid(h // 3, w // 3, device)
    J = field_jacobian(model, sub, px)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    m = {
        "psnr": psnr(warped, sc["clean"]),
        "epe_mean": float(epe.mean()),
        "epe_fg": float(epe[sc["masks"]["foreground"]].mean()),
        "epe_band": float(epe[sc["masks"]["boundary_band"]].mean()),
        "epe_bg": float(epe[sc["masks"]["background"]].mean()),
        "det_min": float(det.min()),
        "folded": float((det < 0).float().mean()),
        "folded_count": int((det < 0).sum()),
        "jacobian_samples": int(det.numel()),
    }
    un = u.detach().cpu().numpy()
    # The field the fit has LEARNED, on the ground truth's own scale so the two
    # are directly comparable: hue is direction, brightness is magnitude.
    vmax = max(1.0, float(sc["ugt_dense"].norm(dim=-1).max()))
    images = {"warped": gray_png(warped),
              "epe": cmap_png(epe.cpu().numpy(), EPE_LUT_MAX),
              "residual": cmap_png((warped - sc["clean"]).abs().cpu().numpy(),
                                   RESID_LUT_MAX),
              "ufit": flow_png(u.reshape(h, w, 2).cpu().numpy(), vmax)}
    grid = {"fit": grid_lines(un, spacing, shape),
            "gt": grid_lines(sc["ugt_dense"].cpu().numpy(), spacing, shape),
            "w": shape[1], "h": shape[0],
            "epe_vmax": max(1.0, float(np.percentile(epe.cpu().numpy(), 99)))}
    return m, images, grid, level_maps(model, shape, device, px)


def train_job(cfg, p, device):
    """One fit, publishing to JOB as it goes. Runs on its own thread."""
    try:
        sc = get_scene(cfg, p["deformation"], p["mismatch"], device)
        shape = sc["shape"]
        h, w = shape
        px = torch.tensor([w, h], device=device, dtype=torch.float32)

        spec = {m["name"]: m for m in resolve_inherits(cfg["models"])}[p["model"]]
        spec = json.loads(json.dumps(spec))                     # deep copy
        if spec["kind"] == "siren":
            spec["siren"].update(width=int(p["width"]),
                                 hidden_layers=int(p["hidden_layers"]),
                                 omega_0=float(p["omega_0"]))
            spec.setdefault("coarse_to_fine", {})
            spec["coarse_to_fine"] = {"enabled": bool(p["coarse_to_fine"]),
                                      "start_levels": 4,
                                      "full_at_step": max(1, int(p["steps"] * 0.5))}
        else:
            g = int(p["grid"])
            spec["grid"] = [g, g]
        spec["output_scale_px"] = float(p["output_scale_px"])

        torch.manual_seed(cfg.get("seed", 0))
        model = build_model(spec, device)
        # n_a is the first layer for a SIREN and the control tensor for the
        # dense one -- the same slot, so the page can name it either way and the
        # two parameterisations stay comparable number for number.
        n_a, n_b = model.n_parameters()
        is_siren = spec["kind"] == "siren"
        fld = model.field if is_siren else None
        n_lv_total = (getattr(model, "n_bands", 16) if is_siren else 0)
        store = "first layer" if is_siren else "control tensor"
        if is_siren:
            f = fld.frequencies()
            note = (f"omega_0 {fld.omega():g}, first-layer waves "
                    f"{float(f.min()):.2f}..{float(f.max()):.2f} cycles across the "
                    f"frame = {w / max(float(f.max()), 1e-6):.0f} px per finest "
                    f"cycle, against a deformation whose finest feature is "
                    f"12-23 px")
        else:
            note = (f"{spec['grid'][0]}x{spec['grid'][1]} control points, "
                    f"{w / spec['grid'][1]:.0f} px apart")

        print(f"[run] {p['deformation']} / {p['mismatch']} / {p['model']}  "
              f"{int(p['steps'])} steps, lr {float(p['lr']):.1e}, "
              f"batch {int(p['batch']):,}, pyramid "
              f"{'on' if int(p.get('pyramid', 1)) else 'off'}  ->  "
              f"{n_a + n_b:,} parameters", flush=True)
        print(f"[params] {n_a + n_b:,} total = {n_a:,} in the {store} "
              f"+ {n_b:,} after it"
              + (f"  ({int(p['width'])} plane waves)" if is_siren else ""),
              flush=True)
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], metrics={"n_parameters": n_a + n_b,
                                          "n_table": n_a, "n_decoder": n_b,
                                          "store": store,
                                          "n_levels_total": n_lv_total}, note=note,
                       images={"source": gray_png(sc["source"]),
                               "target": gray_png(sc["observed"]),
                               # THE GROUND TRUTH ITSELF. Everything else on the
                               # page is a consequence of the deformation; on a
                               # 12 px shear band shown in a 300 px panel the
                               # consequence is a few screen pixels wide and easy
                               # to miss entirely.
},
                       grid={}, stamp=JOB["stamp"] + 1)
        print(f"[images] source and target sent ({w}x{h})", flush=True)
        first_render = True

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps = int(p["steps"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=float(p["lr"]) * 0.03)
        offs = patch_offsets(cfg["training"]["loss"]["lncc"]["window_px"], shape, device)
        n_patch = cfg["training"]["loss"]["lncc"]["n_patches"]
        frac = cfg["training"]["sampling"]["foreground_fraction"]
        reg_n = cfg["training"].get("reg_batch", 8192)
        w_smooth, w_fold = float(p["w_smooth"]), float(p["w_fold"])
        # Without this the loss has a capture radius of about half the LNCC
        # window, ~4 px, against ground-truth displacements of 12-42 px, and no
        # parameterisation can converge. Exposed as a control because it and the
        # level window are substitutes: with neither, the fit lands 43x worse.
        pyr = (cfg["training"].get("image_pyramid", {"sigma_px": [0], "switch_at": [0.0]})
               if int(p.get("pyramid", 1)) else {"sigma_px": [0], "switch_at": [0.0]})
        src_p = build_pyramid(sc["source"], pyr["sigma_px"])
        tgt_p = build_pyramid(sc["observed"], pyr["sigma_px"])
        lvl = -1
        with LOCK:
            JOB["switches"] = [{"step": int(a * int(p["steps"])), "sigma": float(sg)}
                               for a, sg in zip(pyr["switch_at"], pyr["sigma_px"])]
        every = max(1, steps // 40)
        t0 = t0_all = time.perf_counter()

        for step in range(steps + 1):
            if STOP.is_set():
                break
            if is_siren and spec["coarse_to_fine"]["enabled"]:
                a = 4 + (n_lv_total - 4) * min(
                    1.0, step / spec["coarse_to_fine"]["full_at_step"])
                model.set_level_window(a)          # bands here, levels there
                JOB["levels_live"] = round(float(a), 2)
            elif is_siren:
                JOB["levels_live"] = float(n_lv_total)
            if sc["loss"] == "lncc":
                c = sample_points(sc["fg_idx"], n_patch, frac, shape, device)
                xy = (c[:, None, :] + offs[None]).reshape(-1, 2)
            else:
                xy = sample_points(sc["fg_idx"], int(p["batch"]), frac, shape, device)
            new_lvl = pyramid_level(step, steps, pyr["switch_at"])
            if new_lvl != lvl:
                lvl = new_lvl
                with LOCK:
                    JOB["pyramid_sigma"] = pyr["sigma_px"][lvl]
            u = model(xy)
            pred = sample_bilinear(src_p[lvl], xy + u / px)
            gt = sample_bilinear(tgt_p[lvl], xy)
            loss = (lncc_loss(pred, gt, n_patch) if sc["loss"] == "lncc"
                    else ((pred - gt) ** 2).mean())
            photo = loss.detach().item()
            if w_smooth > 0 or w_fold > 0:
                xr = sample_points(sc["fg_idx"], reg_n, frac, shape, device)
                J = field_jacobian(model, xr, px, create_graph=True)
                if w_smooth > 0:
                    loss = loss + w_smooth * ((J - torch.eye(2, device=device)) ** 2
                                              ).sum((1, 2)).mean()
                if w_fold > 0:
                    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                    loss = loss + w_fold * F.relu(0.1 - det).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if step % every == 0 or step == steps:
                m, images, grid, lv = _evaluate(model, sc, device, int(p["grid_spacing"]))
                with LOCK:
                    JOB["levels"] = lv
                    JOB["step"] = step
                    JOB["seconds"] = time.perf_counter() - t0
                    JOB["curve"].append({"step": step, "loss": photo,
                                         "epe_fg": m["epe_fg"],
                                         "epe_band": m["epe_band"],
                                         "epe_bg": m["epe_bg"]})
                    JOB["metrics"] = {**m, "n_parameters": n_a + n_b, "loss": photo,
                                      "n_table": n_a, "n_decoder": n_b,
                                      "store": store,
                                      "loss_kind": sc["loss"],
                                      "n_levels_total": n_lv_total}
                    JOB["images"].update(images)
                    JOB["grid"] = grid
                    JOB["stamp"] += 1
                if first_render:
                    first_render = False
                    print(f"[images] first fit and error map sent at step {step} "
                          f"({time.perf_counter() - t0_all:.1f}s)", flush=True)
    except Exception as e:                                   # surface, don't swallow
        print(f"[run] failed: {type(e).__name__}: {e}", flush=True)
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1
            m, done = JOB["metrics"], JOB["step"] >= JOB["steps"] > 0
        verb = "done " if done else "stopped"
        if "epe_fg" in m:
            print(f"[{verb}] {JOB['seconds']:.1f}s  psnr {m['psnr']:.2f} dB  "
                  f"EPE fg {m['epe_fg']:.3f} / band {m['epe_band']:.3f} / "
                  f"bg {m['epe_bg']:.3f} px  folded {m['folded_count']}/"
                  f"{m['jacobian_samples']}", flush=True)
        else:
            print(f"[{verb}] {JOB['seconds']:.1f}s", flush=True)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>registration &mdash; siren vs control grid</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>registration &mdash; a siren against a control grid</h1>
<p class="sub">The source is warped by a known analytic field to make the target,
so the fit is scored on the <b style="color:#fff">field</b> it recovered, not only on how well the
images line up. Endpoint error is split into the textured foreground, the black
background where nothing constrains the warp, and the band between &mdash; that split
is where the two parameterisations disagree, long after the warped image stops
telling them apart.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is a siren?</button><button onclick="openHelp()">what is this interface?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs_model"></div>
<div class="knobs" id="knobs_train"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="setup" id="setup"></div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_source" width="330" height="460"></canvas>
    <div class="cap">source</div></div>
  <div class="panel"><canvas id="c_target" width="330" height="460"></canvas>
    <div class="cap">target &mdash; source warped, then remapped</div></div>
  <div class="panel"><canvas id="c_warp" width="330" height="460"></canvas>
    <div class="cap">source warped by the fit</div></div>
  <div class="panel"><canvas id="c_levels" width="330" height="460"></canvas>
    <div class="cap">what each band contributes, and how evenly</div></div>
</div>
<div id="levlegend" class="note"></div>
<div class="row equal" style="margin-top:18px">
  <div class="panel"><canvas id="c_grid" width="330" height="460"></canvas>
    <div class="cap">grid &mdash; <i>ground truth</i> vs <b>fit</b></div></div>
  <div class="panel"><canvas id="c_ufit" width="330" height="460"></canvas>
    <div class="cap">the field the fit learned &mdash; hue is direction, brightness
      is magnitude</div></div>
  <div class="panel"><canvas id="c_epe" width="330" height="460"></canvas>
    <div class="cap">endpoint error &mdash; fixed scale 0&ndash;__EPEMAX__ px</div></div>
  <div class="panel"><canvas id="c_resid" width="330" height="460"></canvas>
    <div class="cap">image residual &mdash; fixed scale 0&ndash;__RESMAX__</div></div>
</div>
<div class="row" style="margin-top:18px">
  <div class="panel"><canvas id="c_curve" width="520" height="460"></canvas>
    <div class="cap">endpoint error against the analytic field, by region</div></div>
</div>
<div class="stats" id="stats"></div>
<div class="modal" id="about" onclick="if(event.target===this)closeAbout()">
  <div class="sheet">__ABOUT__</div></div>
<div class="modal" id="help" onclick="if(event.target===this)closeHelp()">
  <div class="sheet"><button class="close" onclick="closeHelp()">close</button>
  __HELP__</div></div>
</div><script>
// A page that fails in the browser but passes every offline check leaves no
// trace at all: the panels are simply black. Report the exception to the server
// so it lands in the terminal next to [run] and [images].
// One-shot breadcrumbs, so "the panels are empty" can be told apart from
// "poll never ran" and from "poll ran and painting did nothing" without a
// browser console.
const _said = {};
function _log(tag, what){
  if (_said[tag]) return;
  _said[tag] = true;
  try { fetch("/api/clientlog?msg=" + encodeURIComponent(tag + ": " + what)); }
  catch (e) {}
}
function _report(what){
  try { fetch("/api/clienterror?msg=" + encodeURIComponent(String(what).slice(0, 800))); }
  catch (e) {}
}
window.onerror = (msg, src, line, col, err) =>
  _report((err && err.stack) || (msg + " @" + line + ":" + col));
window.addEventListener("unhandledrejection", e =>
  _report("unhandled rejection: " + ((e.reason && e.reason.stack) || e.reason)));
function openAbout(){document.getElementById("about").classList.add("open");}
function closeAbout(){document.getElementById("about").classList.remove("open");}
function openHelp(){document.getElementById("help").classList.add("open");}
function closeHelp(){document.getElementById("help").classList.remove("open");}
document.addEventListener("keydown",e=>{
  if(e.key==="Escape"){ closeAbout(); closeHelp(); }});
const SPEC=__SPEC__, KNOBS=__KNOBS__, DEF=__DEF__, LUTMAX=__LUTMAX__;
// slip_band by default: it is the one deformation the smooth parameterisations
// structurally cannot represent, so it separates them on opening rather than
// after you have gone looking.
const sel={deformation: SPEC.deformation.includes("slip_band") ? "slip_band"
                                                              : SPEC.deformation[0],
           mismatch:SPEC.mismatch[0], model:SPEC.model[0]};
const knob=Object.assign({}, DEF);
let LAST=-1, POLL=null, IMG={}, SEEN_RUNNING=false;

const C=document.getElementById("controls");
function group(name, opts, key){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=String(o).replace(/_/g," ");
    b.setAttribute("aria-pressed", sel[key]===o);
    b.onclick=()=>{ sel[key]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      // The L2 data term sits near 1e-4 and 1-LNCC near 3e-1, so the same
      // absolute regulariser weight is ~1000x weaker under LNCC. Follow the
      // loss when the mismatch changes instead of leaving a stale weight.
      if(key==="mismatch"){ const r=SPEC.reg[SPEC.loss[o]];
        knob.w_smooth=r.w_smooth; knob.w_fold=r.w_fold; }
      buildKnobs(); };
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g);
}
group("deformation", SPEC.deformation, "deformation");
group("mismatch", SPEC.mismatch, "mismatch");
group("parameterisation", SPEC.model, "model");
(function(){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label";
  l.textContent="image pyramid";
  const sg=document.createElement("div"); sg.className="seg";
  [["on",1],["off",0]].forEach(([txt,val])=>{
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", knob.pyramid===val);
    b.onclick=()=>{ knob.pyramid=val;
      [...sg.children].forEach(c=>c.setAttribute("aria-pressed", c===b)); };
    sg.appendChild(b); });
  g.append(l,sg); C.appendChild(g);
})();

function isHash(){ return SPEC.kind[sel.model]==="siren"; }
function knobPanel(el, title, list){
  el.innerHTML="";
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  el.appendChild(t);
  list.forEach(p=>{
    if(false && p.name==="interpolation"){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML="<span>interpolation</span>";
      const s=document.createElement("div"); s.className="seg";
      ["linear","smoothstep"].forEach(o=>{
        const b=document.createElement("button"); b.textContent=o;
        b.setAttribute("aria-pressed", knob.interpolation===o);
        b.onclick=()=>{ knob.interpolation=o;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); };
        s.appendChild(b); });
      d.append(lab,s); el.appendChild(d); return;
    }
    if(p.name==="coarse_to_fine"){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML="<span>coarse to fine (level window)</span>";
      const s=document.createElement("div"); s.className="seg";
      [["on",1],["off",0]].forEach(([txt,v])=>{
        const b=document.createElement("button"); b.textContent=txt;
        b.setAttribute("aria-pressed", knob.coarse_to_fine===v);
        b.onclick=()=>{ knob.coarse_to_fine=v;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); };
        s.appendChild(b); });
      d.append(lab,s); el.appendChild(d); return;
    }
    if(knob[p.name]===undefined) knob[p.name]=p.default;
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.log ? (+v).toExponential(1)
                       : (p.step<1 ? (+v).toFixed(3) : String(Math.round(v)));
    const rawOf=v=>p.log ? Math.log10(v) : v;
    const valOf=r=>p.log ? Math.pow(10, r) : +r;
    val.textContent=fmt(knob[p.name]);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=rawOf(p.min); r.max=rawOf(p.max); r.step=p.log?0.02:p.step;
    r.value=rawOf(knob[p.name]);
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${fmt(p.min)}</span><span>${fmt(p.max)}</span>`;
    r.oninput=()=>{ knob[p.name]=valOf(r.value); val.textContent=fmt(knob[p.name]); };
    d.append(lab,r,ends); el.appendChild(d);
  });
}
function buildKnobs(){
  knobPanel(document.getElementById("knobs_model"),
            isHash() ? "encoding" : "control grid",
            isHash() ? KNOBS.siren : KNOBS.control);
  knobPanel(document.getElementById("knobs_train"), "training", KNOBS.train);
}
buildKnobs();

async function startRun(){
  SEEN_RUNNING=false;
  const q=new URLSearchParams(Object.assign({}, sel, knob));
  await fetch("/api/start?"+q);
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 400); poll();
}
document.getElementById("run").onclick=startRun;
document.getElementById("stop").onclick=()=>fetch("/api/stop");

function drawImg(id, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(!src){ g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height); return; }
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  const w=im.width*s, h=im.height*s;
  g.drawImage(im,(cv.width-w)/2,(cv.height-h)/2,w,h);
  const r=cv.getBoundingClientRect();
  _log("paint", `${cv.id} image ${im.width}x${im.height} -> canvas `
               +`${cv.width}x${cv.height}, on screen `
               +`${Math.round(r.width)}x${Math.round(r.height)}`);
}

// Fixed 0-20 level scale, shared with the image page, so a colour means the
// same level whatever L is set to.
function levColor(t){
  // Bright at both ends: viridis' dark end is invisible on a dark image, and the
  // coarse levels -- the ones that carry most of a smooth deformation -- live
  // there. Blue, turquoise, green, amber, red instead.
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
function drawLevels(bk){
  const cv=document.getElementById("c_levels"), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const leg=document.getElementById("levlegend");
  if(!bk || !bk.profile){
    leg.textContent="a control grid has no bands to decompose"; return; }
  const P=bk.profile, pad={l:52,r:12,t:14,b:26};
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const mx=Math.max(...P.map(p=>p.mean))||1, bw=W/P.length;
  g.strokeStyle="#333"; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad.t+H*i/4; g.beginPath();
    g.moveTo(pad.l,y); g.lineTo(pad.l+W,y); g.stroke(); }
  P.forEach((p,i)=>{ const hgt=H*p.mean/mx; g.fillStyle=levColor(p.band/LUTMAX);
    g.fillRect(pad.l+i*bw+1, pad.t+H-hgt, bw-2, hgt); });
  g.strokeStyle="#fff"; g.lineWidth=1.4; g.beginPath();
  P.forEach((p,i)=>{ const x=pad.l+(i+0.5)*bw, y=pad.t+H*(1-Math.min(1,p.top10));
    i ? g.lineTo(x,y) : g.moveTo(x,y); });
  g.stroke();
  g.setLineDash([3,3]); g.strokeStyle="#777"; g.beginPath();
  g.moveTo(pad.l,pad.t+H*0.9); g.lineTo(pad.l+W,pad.t+H*0.9); g.stroke();
  g.setLineDash([]);
  g.fillStyle="#9a9a9a"; g.font="10px sans-serif"; g.textAlign="right";
  g.fillText(mx.toFixed(2)+" px", pad.l-5, pad.t+8);
  g.fillText("0", pad.l-5, pad.t+H);
  g.textAlign="center"; g.fillText("band", pad.l+W/2, cv.height-8);
  g.fillText("0", pad.l+bw/2, cv.height-16);
  g.fillText(String(P.length-1), pad.l+W-bw/2, cv.height-16);
  const sp=P.map(p=>p.top10);
  leg.innerHTML=`<div>bars: mean displacement each band contributes, in pixels, `
    +`on the band colour scale. White line: the share of it sitting in the `
    +`busiest tenth of the frame &mdash; <b>${(Math.min(...sp)*100).toFixed(0)}`
    +`&ndash;${(Math.max(...sp)*100).toFixed(0)}%</b>, against 10% for perfectly `
    +`spread out (dashed). A SIREN band is a plane wave over the whole frame, so `
    +`there is no "which band works here" to report &mdash; the block map this `
    +`replaced named the finest band in 239 of 255 blocks.</div>`;
}
function drawGrid(gr){
  const cv=document.getElementById("c_grid"), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!gr || !gr.gt) return;
  const s=Math.min(cv.width/gr.w, cv.height/gr.h);
  const ox=(cv.width-gr.w*s)/2, oy=(cv.height-gr.h*s)/2;
  const paint=(lines,col,lw,dash)=>{
    g.strokeStyle=col; g.lineWidth=lw; g.setLineDash(dash||[]);
    lines.forEach(L=>{ g.beginPath();
      L.forEach((p,i)=>{ const X=ox+p[0]*s, Y=oy+p[1]*s;
        i?g.lineTo(X,Y):g.moveTo(X,Y); }); g.stroke(); });
    g.setLineDash([]);
  };
  paint(gr.gt, "#e5484d", 1.2);
  paint(gr.fit, "#4da3ff", 1.0, [4,3]);
}

function drawCurve(curve, switches){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const W=cv.width, H=cv.height;
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  if(!curve || curve.length<2) return;
  const pad={l:56,r:14,t:16,b:34};
  const xs=curve.map(c=>c.step);
  const x0=Math.min(...xs), x1=Math.max(...xs)||1;
  // Endpoint error only. The loss is measured against whichever pyramid level
  // is current, so it jumps by two orders of magnitude at a switch while the fit
  // is improving -- on the same axes as a metric that means one thing throughout,
  // it only confuses.
  const vals=curve.flatMap(c=>[c.epe_fg,c.epe_band,c.epe_bg]).filter(v=>v>0);
  const lo=Math.max(1e-8, Math.min(...vals)), hi=Math.max(...vals);
  const X=v=>pad.l+(v-x0)/((x1-x0)||1)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/
                    ((Math.log10(hi)-Math.log10(lo))||1))*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.lineWidth=1;
  for(let d=Math.floor(Math.log10(lo)); d<=Math.ceil(Math.log10(hi)); d++){
    const y=Y(Math.pow(10,d)); if(y<pad.t||y>H-pad.b) continue;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.font="10px monospace";
    g.fillText("1e"+d, 8, y+3);
  }
  const line=(key,col)=>{ g.strokeStyle=col; g.lineWidth=1.8; g.beginPath();
    curve.forEach((c,i)=>{ const X_=X(c.step), Y_=Y(c[key]);
      i?g.lineTo(X_,Y_):g.moveTo(X_,Y_); }); g.stroke(); };
  // The loss is measured against the CURRENT pyramid level, so it is not
  // comparable across a switch: sharpening the target raises the residual even
  // as the geometry improves. Mark the switches rather than leave the jump
  // looking like a divergence.
  (switches || []).forEach(sw=>{
    if(sw.step<=x0 || sw.step>x1) return;
    const X_=X(sw.step);
    g.strokeStyle="#4a4a4a"; g.lineWidth=1; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(X_,pad.t); g.lineTo(X_,H-pad.b); g.stroke();
    g.setLineDash([]);
    g.fillStyle="#7a7a7a"; g.font="9px sans-serif"; g.textAlign="center";
    g.fillText(`\u03c3 ${sw.sigma}`, X_, pad.t-3); g.textAlign="left";
  });
  line("epe_fg","#4da3ff"); line("epe_band","#cf6bd6"); line("epe_bg","#e5a23c");
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  g.fillText("iteration", W/2-22, H-10);
  [["endpoint error, foreground (px)","#4da3ff"],
   ["boundary band (px)","#cf6bd6"],
   ["background (px)","#e5a23c"]].forEach(([t,c],i)=>{
    g.fillStyle=c; g.fillRect(W-pad.r-160, pad.t+i*15-8, 9, 2);
    g.fillStyle="#9a9a9a"; g.fillText(t, W-pad.r-145, pad.t+i*15-3); });
}

function setupLine(r){
  const m=r.metrics||{};
  const loss=SPEC.loss[sel.mismatch]||"l2";
  let t=`<b>${sel.deformation.replace(/_/g," ")}</b> <span class="dim">warp</span>`
      +` &nbsp;&middot;&nbsp; <b>${sel.mismatch.replace(/_/g," ")}</b>`
      +` <span class="dim">intensity, ${loss} loss</span>`
      +` &nbsp;&middot;&nbsp; <b>${sel.model}</b>`;
  // white, not dim: the parameter count is a setting like the others above it,
  // decided before the first step, and not a running metric.
  if(m.n_parameters!==undefined){
    t+=` &nbsp;&middot;&nbsp; <span style="color:#fff">`
      +`${m.n_parameters.toLocaleString()} parameters`;
    if(m.n_table!==undefined)
      t+=`, ${m.n_table.toLocaleString()} in the ${m.store} `
        +`+ ${m.n_decoder.toLocaleString()} in the decoder`;
    t+=`</span>`;
  }
  if(r.note) t+=`<br><span class="dim">${r.note}</span>`;
  document.getElementById("setup").innerHTML=t;
}
async function poll(){
  const r=await (await fetch("/api/state")).json();
  _log("poll", `stamp=${r.stamp} step=${r.step} running=${r.running} `
              +`images=[${Object.keys(r.images || {})}] `
              +`grid=${r.grid && r.grid.gt ? "yes" : "no"}`);
  setupLine(r);
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";

  if(r.stamp!==LAST){
    LAST=r.stamp;
    drawImg("c_source", r.images.source); drawImg("c_target", r.images.target);
    drawImg("c_warp", r.images.warped);   drawImg("c_epe", r.images.epe);
    drawImg("c_ufit", r.images.ufit); drawImg("c_resid", r.images.residual);
    drawGrid(r.grid); drawLevels(r.levels);
    drawCurve(r.curve, r.switches); stats(r);
  }
  // Only stop once the run has actually been observed running: a not-yet-started
  // job and a finished one look identical from here.
  if(r.running) SEEN_RUNNING=true;
  if(SEEN_RUNNING && !r.running && POLL){ clearInterval(POLL); POLL=null; }
}
function stats(r){
  const m=r.metrics||{};
  if(m.psnr===undefined){ document.getElementById("stats").innerHTML=
    r.running ? "running&hellip;" : ""; return; }
  const fold=m.folded_count>0
    ? `<span class="bad">${m.folded_count} of ${m.jacobian_samples} samples</span>`
    : `<b>none</b> of ${m.jacobian_samples} samples`;
  document.getElementById("stats").innerHTML=
    `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `+
    `${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; `+
    `${m.loss_kind} loss <b>${m.loss.toExponential(2)}</b>`+
    (r.pyramid_sigma>0 ? ` &nbsp;&middot;&nbsp; pyramid sigma <b>${r.pyramid_sigma}</b> px` : "")+
    (r.levels_live!==null ? ` &nbsp;&middot;&nbsp; levels live <b>${r.levels_live}</b>`
                            +` of ${m.n_levels_total||""}` : "")+
    `<br>`+
    `endpoint error &nbsp; foreground <b>${m.epe_fg.toFixed(3)}</b> px`+
    ` &nbsp; boundary band <b>${m.epe_band.toFixed(3)}</b>`+
    ` &nbsp; background <b>${m.epe_bg.toFixed(3)}</b><br>`+
    `psnr vs the clean warp <b>${m.psnr.toFixed(2)}</b> dB`+
    ` &nbsp;&middot;&nbsp; min det J <b>${m.det_min.toFixed(3)}</b>`+
    ` &nbsp; folded ${fold}`+
    ` &nbsp;&middot;&nbsp; <b>${m.n_parameters.toLocaleString()}</b> parameters`;
}
// Open on a running fit rather than an empty page: the default configuration is
// the one worth seeing first, and it costs one keystroke to stop it.
setupLine({});
startRun();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    cfg = None
    device = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page is generated per request and changes whenever the script
        # does; a cached copy is indistinguishable from a broken one.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            cfg = self.cfg
            resolved = resolve_inherits(cfg["models"])
            models = [m["name"] for m in resolved]
            kinds = {m["name"]: m["kind"] for m in resolved}
            sm = cfg["training"]["loss"]["smoothness"]["weight"]
            fo = cfg["training"]["loss"]["folding"]["weight"]
            spec = {"deformation": [d["name"] for d in cfg["deformations"]],
                    "mismatch": [m["name"] for m in cfg["modality_mismatch"]],
                    "model": models, "kind": kinds,
                    "loss": {m["name"]: m.get("loss", "l2")
                             for m in cfg["modality_mismatch"]},
                    "reg": {k: {"w_smooth": sm[k] if isinstance(sm, dict) else sm,
                                "w_fold": fo[k] if isinstance(fo, dict) else fo}
                            for k in ("l2", "lncc")}}
            page = (PAGE.replace("__CSS__", CSS)
                        .replace("__ABOUT__", ABOUT_HTML)
                        .replace("__HELP__", INTERFACE_REG)
                        .replace("__SPEC__", json.dumps(spec))
                        .replace("__KNOBS__", json.dumps(KNOB_SPEC))
                        .replace("__DEF__", json.dumps(KNOB_DEFAULTS))
                        .replace("__LUTMAX__", str(LEVEL_LUT_MAX))
                        .replace("__EPEMAX__", f"{EPE_LUT_MAX:g}")
                        .replace("__RESMAX__", f"{RESID_LUT_MAX:g}"))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            # RUNNING FROM THE MOMENT THE JOB IS ACCEPTED, not from the moment the
            # worker gets going. The thread has to build the scene first, and a poll
            # that lands in that window used to see running=false and cancel itself,
            # so the page waited forever on a fit that was running fine.
            with LOCK:
                JOB["running"] = True
                JOB["stamp"] += 1
            p = dict(KNOB_DEFAULTS)
            p.update({k: (float(v) if _isnum(v) else v) for k, v in q.items()})
            threading.Thread(target=train_job, args=(self.cfg, p, self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clientlog":
            print(f"[browser] {q.get('msg', '')}", flush=True)
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clienterror":
            print(f"[client] {q.get('msg', '')}", flush=True)
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            if JOB["running"]:
                print("[stop] requested", flush=True)
            STOP.set()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/state":
            with LOCK:
                return self._send(json.dumps(JOB), "application/json")
        self.send_error(404)


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


KNOB_SPEC = {
    "siren": [
        {"name": "width", "label": "width (plane waves)", "min": 32, "max": 1024,
         "default": 256, "step": 32},
        {"name": "hidden_layers", "label": "hidden sine layers", "min": 1,
         "max": 8, "default": 3, "step": 1},
        # Set against the DEFORMATION's scale, not the image's. The finest thing
        # in these warps is a 12 px shear band on a 904 px frame, which is ~75
        # cycles; omega_0 = 30 starts the first layer three decades under that
        # and lets the layers above build the rest.
        {"name": "omega_0", "label": "omega_0", "min": 5, "max": 500,
         "default": 30, "step": 1, "log": True},
        {"name": "coarse_to_fine"},
    ],
    "control": [
        {"name": "grid", "label": "control points per axis", "min": 4,
         "max": 512, "default": 16, "step": 4},
    ],
    "train": [
        # A SIREN's step, not a table's: every weight here touches the whole
        # field, so the same 5e-4 the fitting page measured.
        {"name": "lr", "label": "learning rate", "min": 1e-6, "max": 3e-1,
         "default": 5e-4, "step": 1e-6, "log": True},
        {"name": "steps", "label": "iterations", "min": 100, "max": 6000,
         "default": 1500, "step": 100},
        {"name": "batch", "label": "batch size (sample points)", "min": 4096,
         "max": 262144, "default": 65536, "step": 4096},
        {"name": "w_smooth", "label": "smoothness weight", "min": 1e-6,
         "max": 1e-1, "default": 1e-3, "step": 1e-6, "log": True},
        {"name": "w_fold", "label": "folding penalty weight", "min": 1e-6,
         "max": 1.0, "default": 1e-2, "step": 1e-6, "log": True},
        {"name": "output_scale_px", "label": "displacement scale (px)", "min": 4,
         "max": 160, "default": 40, "step": 4},
        {"name": "grid_spacing", "label": "overlay grid spacing (px)", "min": 8,
         "max": 128, "default": 32, "step": 8},
    ],
}
KNOB_DEFAULTS = {k["name"]: k.get("default") for g in KNOB_SPEC.values() for k in g
                 if "default" in k}
KNOB_DEFAULTS.update(coarse_to_fine=1, pyramid=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config/registration_benchmark.yaml"))
    p.add_argument("--port", type=int, default=8121)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    Handler.cfg = cfg
    Handler.device = torch.device(a.device)
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except OSError as e:
        if e.errno == 98:
            sys.exit(f"port {a.port} is already in use -- either this server is "
                     f"already running (open http://localhost:{a.port}) or pass "
                     f"--port with a free one")
        raise
    print(f"http://localhost:{a.port}   (device {a.device})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
