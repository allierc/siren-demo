#!/usr/bin/env python
"""Browser GUI for stage 1: fit the painting with a SIREN, and watch its scales.

    python scripts/siren_demo/gui_image.py      # http://localhost:8122

The twin of ngp-demo's page, for a representation that has no grid: a SIREN is a
plain MLP whose activations are sines, and the detail lives in the frequencies
rather than in a table.  Those frequencies are not hidden, though.  The first
layer computes sin(omega_0 * (w_i . x + b_i)) per unit, which is a plane wave of
omega_0 |w_i| / 2pi cycles across the picture, so sorting the units by |w_i|
gives a ladder of scales that can be read, gated and drawn exactly the way the
hash grid's levels are.

Two differences from that page, both real: the ladder here is LEARNED, so it
moves while the fit runs, and gating a band is not a clean spectral cut, because
every later layer mixes what it is given.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren import SirenField
from siren.utils import BilinearImage, pixel_centers, psnr, read_image, render
from PIL import Image, ImageDraw

from siren.webui import ABOUT_HTML, CSS, INTERFACE_IMAGE, cmap_png, gray_png,\
    png_data_uri, signed_rgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(ROOT, "assets/girl_with_a_pearl_earring.jpg")
# Band colours come from a fixed 0..15 scale: there are always 16 bands, so a
# colour means the same band in every run and two settings compare by eye.
N_BANDS = 16
LEVEL_LUT_MAX = N_BANDS - 1
# The error panel is on a fixed scale too: auto-scaling to each refresh's own
# 99th percentile made the panel brighten as the fit improved, which reads as
# the opposite of what happened.
ERROR_LUT_MAX = 0.10

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "ladder": [], "note": "", "stamp": 0,
       "history": [], "blocks": {}, "decomp_mode": "adds"}
LOCK = threading.Lock()
STOP = threading.Event()
IMAGES = {}                          # downsample factor -> (tensor, target, coords)


def get_image(path, down, device):
    # Keyed by device as well: /api/preview asks for the shape on the CPU, and a
    # cache hit from that would hand the CUDA training loop CPU tensors.
    key = (path, down, str(device))
    if key in IMAGES:
        return IMAGES[key]
    ref = read_image(path)
    t = torch.from_numpy(ref).to(device)
    if down > 1:
        t = F.avg_pool2d(t.permute(2, 0, 1)[None], down)[0].permute(1, 2, 0)
    arr = t.cpu().numpy()
    target = BilinearImage(arr, device)
    coords = pixel_centers(arr.shape[0], arr.shape[1], device)
    IMAGES[key] = (t, target, coords, arr.shape)
    return IMAGES[key]


SHAPES = {}


def image_shape(path, down):
    """(h, w, c) after downsampling, without building any tensor."""
    key = (path, down)
    if key not in SHAPES:
        h, w, c = read_image(path).shape
        SHAPES[key] = (h // down, w // down, c)
    return SHAPES[key]


def build(p, shape):
    """The SirenField a given set of controls asks for, plus its frequency ladder."""
    h, w, c = shape
    kwargs = dict(
        n_input_dims=2, n_output_dims=c, output_activation="sigmoid",
        width=int(p["width"]), hidden_layers=int(p["hidden_layers"]),
        omega_0=float(p["omega_0"]),
        outermost_linear=str(p["outermost_linear"]) in ("1", "1.0", "True", "linear"),
        learnable_omega=str(p["learnable_omega"]) in ("1", "1.0", "True", "on"))
    model = SirenField(**kwargs)
    return model, ladder_of(model, w)


def ladder_of(model, w):
    """The 16 bands, from the first layer's own weights.

    The hash grid's ladder is a setting; this one is a measurement, and it moves
    while the fit runs.
    """
    f = model.frequencies().cpu()
    band = model.band_of(N_BANDS).cpu()
    out = []
    for b in range(N_BANDS):
        fb = f[band == b]
        if not len(fb):
            continue
        out.append({"band": b, "units": int(len(fb)),
                    "f_lo": round(float(fb.min()), 3),
                    "f_hi": round(float(fb.max()), 3),
                    "px": round(w / max(float(fb.mean()), 1e-6), 2)})
    return out


def describe(model, shape):
    h, w, c = shape
    n_first, n_rest = model.n_parameters()
    f = model.frequencies()
    return {"n_first": n_first, "n_rest": n_rest, "n_total": n_first + n_rest,
            "n_values": h * w * c,
            "fraction_of_values": (n_first + n_rest) / (h * w * c),
            "n_bands": N_BANDS, "channels": c,
            "omega": model.omega(),
            "outermost_linear": isinstance(model.net.net[-1], torch.nn.Linear),
            "learnable": bool(model.net.learnable_omega),
            "f_min": round(float(f.min()), 3), "f_max": round(float(f.max()), 3),
            "finest_px_per_cycle": w / max(float(f.max()), 1e-6),
            "width": w, "height": h}


@torch.no_grad()
def band_maps(model, shape, device, block_px=64, sub=3, thresh=0.08):
    """Where each frequency band does its work.

    Renders the image with bands 0..k enabled for every k and differences
    consecutive renders, so `deltas[b]` is how much the picture changes when band
    b is released.  The layers after the first are nonlinear and mix, so this is
    the marginal effect of a band given the coarser ones, not a term of a linear
    decomposition -- but that marginal is exactly what "this band is doing the
    work here" means.

    Returns a per-pixel effective band (amplitude-weighted mean) and, per block,
    the band that dominates it, which is what the overlay draws a wavelength grid
    for.

    Evaluated on a grid `sub` times coarser than the image: this costs 17 full
    renders, and at full resolution that is far more than the fit itself.
    """
    h, w, c = shape
    hs, ws = max(8, h // sub), max(8, w // sub)
    coords = pixel_centers(hs, ws, device)
    bs = max(2, block_px // sub)
    lad = ladder_of(model, w)
    px_of = {L["band"]: L["px"] for L in lad}
    prev, deltas = None, []
    for k in range(N_BANDS + 1):
        model.set_band_window(float(k), N_BANDS)
        out = render(model, coords, (hs, ws, c))
        if prev is not None:
            deltas.append((out - prev).abs().mean(-1))
        prev = out
    model.set_band_window(float(N_BANDS), N_BANDS)
    D = torch.stack(deltas)                                   # (B, H, W)

    lev = torch.arange(D.shape[0], device=D.device, dtype=D.dtype)
    eff = (D * lev[:, None, None]).sum(0) / D.sum(0).clamp(min=1e-9)

    # WHAT EACH BAND CONTRIBUTES, AND HOW EVENLY -- not "which band works in
    # this block", which is the question ngp-demo's page asks and which has no
    # answer here.  A grid level only touches its own cells, so asking a block
    # which level moved it is a real question.  A SIREN band is a plane wave
    # across the whole frame, and it shows: measured on this fit, the top 10% of
    # blocks hold only 18-33% of any band's total displacement (10% would be
    # perfectly uniform), and every band's amplitude is within a factor of four
    # of every other's.  The block version therefore returned band 15 in 255 of
    # 255 blocks -- true, useless.  So the panel reports the profile instead.
    nb = F.avg_pool2d(D[None], bs, stride=bs, ceil_mode=True)[0]      # (B, Hb, Wb)
    flat = nb.reshape(nb.shape[0], -1)
    k = max(1, int(0.1 * flat.shape[1]))
    top = torch.topk(flat, k, dim=1).values.sum(1) / flat.sum(1).clamp(min=1e-12)
    profile = [{"band": b, "mean": float(flat[b].mean()),
                "top10": float(top[b])} for b in range(flat.shape[0])]
    return eff.cpu().numpy(), profile


# The decomposition window keeps a handle on the fit so it can be re-rendered
# after the run ends.  It is deep-copied before its unit gains are touched: the
# gains live on the model the trainer is still stepping.
MODEL = {"model": None, "shape": None}
# Which view the LIVE panel draws.  One mode, not all three: the montage travels
# in the state payload on every poll, and three of them would be three times the
# bytes for two views nobody is looking at.
DECOMP = {"mode": "adds", "side": 110}
# What band b adds, on a fixed symmetric scale, so a panel darkens as the band's
# contribution shrinks instead of rescaling to its own maximum.
DECOMP_LUT_MAX = 0.10


@torch.no_grad()
def decompose(model, shape, device, mode="adds", n_panels=N_BANDS, side=420):
    """Render the frequency bands one at a time.

    A band is isolated by the same gain vector the coarse-to-fine window uses --
    `unit_gain`, one entry per first-layer unit -- so there is no second forward
    path and nothing here is reimplemented.

    mode="alone"  the rest of the network fed by that band and nothing else
        "upto"    bands 0..b, the fit as it stands at that scale
        "adds"    upto(b) - upto(b-1), what releasing the band changed, signed.
                  The first tile is the BASELINE those differences start from:
                  with every unit gated off the first layer emits zeros and the
                  rest of the network returns a constant, which is not black, so
                  without that tile the montage is a decomposition of nothing.
    """
    h, w, c = shape
    sub = max(1, int(round(max(h, w) / side)))
    hs, ws = max(8, h // sub), max(8, w // sub)
    keep = model.unit_gain.clone()
    lad = {L["band"]: L for L in ladder_of(model, w)}
    bands = list(range(N_BANDS))[-(n_panels - 1 if mode == "adds" else n_panels):]
    cache, panels = {}, []

    def grid_for(b):
        """Every tile at the tile resolution -- NOT at its own band's Nyquist.

        ngp-demo samples each level at its own nodes, because a grid level truly
        cannot represent anything finer than its lattice.  The same rule here
        would be wrong, and measurably so: gating a band changes what every later
        layer sees, and those layers manufacture harmonics, so the difference a
        band makes is mostly finer than the band itself.  Measured on the default
        fit, the fraction of a tile's spectral energy above that band's own top
        frequency is 70% at band 1, 90% at band 5, 93% at band 10 and 63% at band
        15.  Sampling at 2 points per cycle would have thrown all of that away.
        """
        return hs, ws

    def upto(k, ny, nx):
        key = (k, ny, nx)
        if key not in cache:
            g = torch.zeros_like(model.unit_gain)
            if k >= 0:
                band = model.band_of(N_BANDS)
                g[band <= k] = 1.0
            model.set_unit_gain(g)
            cache[key] = render(model, pixel_centers(ny, nx, device),
                                (ny, nx, c)).clamp(0, 1)
        return cache[key]

    def blow_up(rgb):
        if rgb.shape[:2] == (hs, ws):
            return rgb
        return np.asarray(Image.fromarray(rgb).resize((ws, hs), Image.NEAREST))

    def meta(b, kind, amp, ny, nx):
        L = lad.get(b, {"units": 0, "f_lo": 0.0, "f_hi": 0.0, "px": float(w)})
        return {"level": b, "label": (f"0..{b}" if kind == "base" else f"B{b}"),
                "cells": L["units"], "px_per_cell": L["px"],
                "f_lo": L["f_lo"], "f_hi": L["f_hi"], "dense": False,
                "amp": amp, "kind": kind, "grid": f"{nx}x{ny}"}

    try:
        if mode == "adds" and bands[0] > 0:
            b0 = bands[0] - 1
            ny, nx = grid_for(b0)
            base = upto(b0, ny, nx)
            panels.append({**meta(b0, "base", float(base.mean()), ny, nx),
                           "rgb": blow_up(_u8(base))})
        for b in bands:
            ny, nx = grid_for(b)
            if mode == "upto":
                img = upto(b, ny, nx)
                rgb, amp = _u8(img), float(img.std())
            elif mode == "adds":
                # SIGNED, not |.|: a band that darkens a region and one that
                # brightens it are doing opposite things.
                d = (upto(b, ny, nx) - upto(b - 1, ny, nx)).mean(-1)
                rgb, amp = signed_rgb(d.cpu().numpy(), DECOMP_LUT_MAX), \
                    float(d.abs().mean())
            else:
                g = torch.zeros_like(model.unit_gain)
                g[model.band_of(N_BANDS) == b] = 1.0
                model.set_unit_gain(g)
                img = render(model, pixel_centers(ny, nx, device),
                             (ny, nx, c)).clamp(0, 1)
                rgb, amp = _u8(img), float(img.std())
            panels.append({**meta(b, mode, amp, ny, nx), "rgb": blow_up(rgb)})
    finally:
        model.set_unit_gain(keep)
    return {"panels": panels, "mode": mode, "n_levels": N_BANDS,
            "render": f"{ws}x{hs}"}


def _u8(t):
    return (np.clip(t.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)


def montage_png(out, cols=4, gap=2, label=True):
    """The decomposition as ONE picture, so the live panel is an image like any
    other and not a second layout to keep in step with the first."""
    tiles = [p["rgb"] for p in out["panels"]]
    th, tw, _ = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * (th + gap) - gap, cols * (tw + gap) - gap, 3), np.uint8)
    for i, t in enumerate(tiles):
        y, x = (i // cols) * (th + gap), (i % cols) * (tw + gap)
        sheet[y:y + th, x:x + tw] = t
    im = Image.fromarray(sheet)
    if label:
        d = ImageDraw.Draw(im)
        for i, p in enumerate(out["panels"]):
            y, x = (i // cols) * (th + gap), (i % cols) * (tw + gap)
            d.text((x + 2, y + 1), p["label"], fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def train_job(p, device):
    try:
        down = max(1, int(p["downsample"]))
        img, target, coords, shape = get_image(DEFAULT_IMAGE, down, device)
        h, w, c = shape
        torch.manual_seed(0)
        model, ladder = build(p, shape)
        model = model.to(device)
        MODEL["model"], MODEL["shape"] = model, shape
        info = describe(model, shape)
        label = (f"w{int(p['width'])} x{int(p['hidden_layers'])} "
                 f"om{float(p['omega_0']):g}")
        print(f"[run] width {int(p['width'])} x {int(p['hidden_layers'])} hidden "
              f"layers, omega_0 {float(p['omega_0']):g}"
              f"{' (learnable)' if info['learnable'] else ''}, "
              f"{'linear' if info['outermost_linear'] else 'sine'} output  "
              f"{int(p['steps'])} steps, lr {float(p['lr']):.1e}, "
              f"batch {int(p['batch']):,}  -> {info['n_total']:,} params "
              f"({info['fraction_of_values']*100:.1f}% of the image)", flush=True)
        print(f"[params] {info['n_total']:,} total = {info['n_first']:,} in the "
              f"first layer ({int(p['width'])} plane waves) + {info['n_rest']:,} "
              f"after it", flush=True)
        print(f"[bands] first-layer frequencies {info['f_min']:.2f}..{info['f_max']:.2f} "
              f"cycles across the picture = {w / max(info['f_max'], 1e-6):.1f} px "
              f"per finest cycle", flush=True)
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], ladder=ladder, metrics=info,
                       note=f"{w}x{h}x{c}, {info['n_total']:,} parameters "
                            f"({info['n_first']:,} in the first layer, "
                            f"{info['fraction_of_values']*100:.1f}% of the "
                            f"{info['n_values']:,} reference values), "
                            f"omega_0 {info['omega']:g}, "
                            f"{info['f_min']:.2f}..{info['f_max']:.2f} cycles, "
                            f"{info['finest_px_per_cycle']:.2f} px per finest cycle",
                       images={"reference": gray_png(img)}, stamp=JOB["stamp"] + 1)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps = int(p["steps"])
        batch = int(p["batch"])
        every = max(1, steps // 40)
        ref_t = target(coords).reshape(h, w, c)
        t_train = 0.0
        for step in range(steps + 1):
            if STOP.is_set():
                break
            t0 = time.perf_counter()
            xy = torch.rand(batch, 2, device=device)
            pred = model(xy)
            with torch.no_grad():
                gt = target(xy)
            if p["loss"] == "relative_l2":
                loss = ((pred - gt) ** 2 / (pred.detach() ** 2 + 1e-2)).mean()
            else:
                loss = ((pred - gt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_train += time.perf_counter() - t0

            if step % every == 0 or step == steps:
                fit = render(model, coords, (h, w, c)).clamp(0, 1)
                db = psnr(fit, ref_t)
                err = (fit - ref_t).abs().mean(-1).cpu().numpy()
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = t_train
                    JOB["curve"].append({"step": step, "t": t_train, "psnr": db,
                                         "loss": loss.detach().item()})
                    JOB["metrics"] = {**info, **describe(model, shape),
                                      "psnr": db, "loss": loss.detach().item()}
                    JOB["images"]["fit"] = gray_png(fit)
                    JOB["images"]["error"] = cmap_png(err, ERROR_LUT_MAX)
                    JOB["stamp"] += 1
                eff, profile = band_maps(model, shape, device)
                # The same decomposition the decompose window draws, at a tile
                # size that costs one small render per band rather than one full
                # one -- what makes it affordable on every refresh.
                dec = decompose(model, shape, device, mode=DECOMP["mode"],
                                side=DECOMP["side"])
                with LOCK:
                    JOB["images"]["levels"] = cmap_png(eff, LEVEL_LUT_MAX, "levels")
                    JOB["images"]["montage"] = montage_png(dec)
                    JOB["decomp_mode"] = dec["mode"]
                    JOB["ladder"] = ladder_of(model, w)   # it MOVES while training
                    JOB["blocks"] = {"profile": profile, "n_levels": N_BANDS}
                    JOB["stamp"] += 1
        with LOCK:
            if JOB["curve"]:
                JOB["history"].append({
                    "label": label, "params": info["n_total"],
                    "psnr": JOB["curve"][-1]["psnr"], "seconds": t_train,
                    "curve": [{"t": q["t"], "psnr": q["psnr"]} for q in JOB["curve"]]})
                JOB["history"] = JOB["history"][-8:]
    except Exception as e:
        print(f"[run] failed: {type(e).__name__}: {e}", flush=True)
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1
            m, done = JOB["metrics"], JOB["step"] >= JOB["steps"] > 0
        verb = "done " if done else "stopped"
        print(f"[{verb}] {JOB['seconds']:.1f}s  "
              + (f"psnr {m['psnr']:.2f} dB" if "psnr" in m else ""), flush=True)


# A SIREN has three: how wide, how deep, and omega_0.  The paper fixes the init
# from those, and everything else about the network follows.  Two switches sit
# beside them because the reference implementation exposes them and they change
# what the fit is: a linear output layer instead of one more sine, and a
# trainable omega (which the connectome-gnn copy added, with an L2 to hold it).
FIXED = {"loss": "l2"}
KNOBS = [
    {"name": "width", "label": "width (units per layer)", "min": 32, "max": 1024,
     "default": 256, "step": 32},
    {"name": "hidden_layers", "label": "hidden sine layers", "min": 1, "max": 8,
     "default": 3, "step": 1},
    # THE knob. The first layer starts at omega_0 |w_i| / 2pi cycles across the
    # picture, and with |w_i| ~ 1/d at init that is roughly omega_0 / 12 cycles
    # -- so 30 (the paper's default) puts every starting wave below 3 cycles on
    # a 904 px image, and the deeper layers have to manufacture everything
    # finer.  Log scale, because the useful range spans two decades.
    {"name": "omega_0", "label": "omega_0 (first-layer frequency)", "min": 5,
     "max": 2000, "default": 60, "step": 1, "log": True},
]
TRAIN_KNOBS = [
    # Two orders of magnitude below the hash grid's: the table's entries are
    # local and a big step only moves a few of them, while every weight here
    # touches the whole picture.
    # 5e-4 at omega_0 = 60 is the best cell of a measured 6x4 sweep (29.98 dB at
    # 400 steps); the product omega_0*lr is what breaks, and 0.03 is safe.
    {"name": "lr", "label": "learning rate", "min": 1e-6, "max": 1e-2,
     "default": 5e-4, "step": 1e-6, "log": True},
    {"name": "steps", "label": "iterations", "min": 100, "max": 8000,
     "default": 1500, "step": 100},
    {"name": "batch", "label": "batch size (random pixels)", "min": 4096,
     "max": 1048576, "default": 262144, "step": 4096},
]
DEFAULTS = {k["name"]: k["default"] for k in KNOBS + TRAIN_KNOBS}
DEFAULTS.update(FIXED, downsample=2, outermost_linear=1, learnable_omega=0)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>fitting the painting</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>siren &mdash; fitting a painting</h1>
<p class="sub">Random pixel coordinates in, RGB out, through a plain MLP whose
activations are sines. There is no grid and no table: the scales live in the
frequencies, and the first layer wears them openly &mdash; unit <i>i</i> computes
sin(<b>&omega;<sub>0</sub></b>(w<sub>i</sub>&middot;x + b<sub>i</sub>)), a plane wave of
<b style="color:#e5a23c">&omega;<sub>0</sub>|w<sub>i</sub>|/2&pi; cycles</b> across the
picture. Sorted by that, the units make a ladder, and the ladder can be gated, drawn
and taken apart. It is a <i>learned</i> ladder, so watch it move while the fit
runs.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is an ngp?</button><button onclick="openHelp()">what is this interface?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs_enc"></div>
<div class="knobs" id="knobs_train"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button>
  <button id="clear">clear history</button>
  <button id="decompose">decompose</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="note" id="note"></div>
<div class="row equal" style="margin-top:18px">
  <div class="panel"><canvas id="c_ref" width="330" height="460"></canvas>
    <div class="cap">reference</div></div>
  <div class="panel"><canvas id="c_fit" width="330" height="460"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_err" width="330" height="460"></canvas>
    <div class="cap">absolute error &mdash; fixed scale 0&ndash;__ERRMAX__</div></div>
  <div class="panel"><canvas id="c_levels" width="330" height="460"></canvas>
    <div class="cap">what each band contributes, and how evenly</div></div>
</div>
<div id="zoomnote" class="note"></div>
<div id="levlegend" class="note"></div>
<div class="row grid4" style="margin-top:18px">
  <div class="panel span2"><canvas id="c_curve" width="700" height="460"></canvas>
    <div class="cap">psnr against training time &mdash; this run and the last few</div></div>
  <div class="panel"><canvas id="c_effmap" width="330" height="460"></canvas>
    <div class="cap">effective band per pixel &mdash; same 0&ndash;15 scale</div></div>
  <div class="panel"><canvas id="c_montage" width="330" height="460"></canvas>
    <div class="cap" id="cap_mont">the 16 frequency bands, one at a time</div>
    <div class="seg" id="decmodes" style="margin-top:6px"></div></div>
</div>
<div class="row" style="margin-top:20px">
  <div class="panel"><div class="label">frequency ladder &mdash; measured, and moving</div>
    <div id="ladder"></div></div>
  <div class="panel"><div class="label">runs so far</div>
    <div id="history"></div></div>
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
const KNOBS=__KNOBS__, TRAIN=__TRAIN__, DEF=__DEF__, LUTMAX=__LUTMAX__;
const knob=Object.assign({}, DEF);
let LAST=-1, POLL=null, IMG={}, SEEN_RUNNING=false;
// Declared up here, not beside the drawing code: the magnifier toggle below is
// built while the controls are, and a `const` referenced before its declaration
// is a ReferenceError that kills the whole script rather than one handler.
const ZOOM={on:false, u:0.5, v:0.5, f:4, refFixed:true};
const PANELS=["c_ref","c_fit","c_err","c_levels"];
let LASTBLOCKS=null;
// White line, with the compression figure carrying the verdict: green under
// 50% of the image's own values, amber under 100%, red once the "compression"
// is an expansion.
function noteHTML(m){
  if(!m || m.n_total===undefined) return "";
  const f=m.fraction_of_values*100;
  const col = f<50 ? "#2ea043" : (f<100 ? "#e5a23c" : "#e5484d");
  return `<span style="color:#fff">${m.width}&times;${m.height}&times;${m.channels}, `
       + `${m.n_total.toLocaleString()} parameters `
       + `(${m.n_first.toLocaleString()} in the first layer `
       + `+ ${m.n_rest.toLocaleString()} after it, `
       + `<b style="color:${col}">${f.toFixed(1)}%</b> of the `
       + `${m.n_values.toLocaleString()} reference values), `
       + `&omega;<sub>0</sub> ${m.omega.toFixed(0)}, `
       + `${m.f_min.toFixed(2)}&ndash;${m.f_max.toFixed(2)} cycles, `
       + `${m.finest_px_per_cycle.toFixed(1)} px per finest cycle</span>`;
}

const C=document.getElementById("controls");
function seg(name, opts, key, after){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=String(o).replace(/_/g," ");
    b.setAttribute("aria-pressed", knob[key]===o);
    b.onclick=()=>{ knob[key]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      if(after) after(); preview(); };
    s.appendChild(b); });
  g.append(l,s); C.appendChild(g);
}
seg("downsample", [1,2,4], "downsample");
// Two switches from the reference implementation, both of which change what the
// network is rather than how it is trained.
(function(){
  [["output layer", [["linear",1],["one more sine",0]], "outermost_linear"],
   ["omega_0", [["fixed",0],["learnable",1]], "learnable_omega"]].forEach(
  ([title, opts, key])=>{
    const g=document.createElement("div"); g.className="group";
    const l=document.createElement("div"); l.className="label"; l.textContent=title;
    const sg=document.createElement("div"); sg.className="seg";
    opts.forEach(([txt,val])=>{
      const b=document.createElement("button"); b.textContent=txt;
      b.setAttribute("aria-pressed", knob[key]===val);
      b.onclick=()=>{ knob[key]=val;
        [...sg.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
        preview(); };
      sg.appendChild(b); });
    g.append(l,sg); C.appendChild(g); });
})();
// Magnifier mode is a view setting, not a model setting: it must not restart a
// fit, so it is handled here rather than through the knob object.
(function(){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent="magnifier";
  const sg=document.createElement("div"); sg.className="seg";
  [["reference fixed",true],["all panels",false]].forEach(([txt,val])=>{
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", ZOOM.refFixed===val);
    b.onclick=()=>{ ZOOM.refFixed=val;
      [...sg.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      redrawAll(); zoomNote(); };
    sg.appendChild(b); });
  g.append(l,sg); C.appendChild(g);
})();

function panel(el, title, list){
  el.innerHTML="";
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  el.appendChild(t);
  list.forEach(p=>{
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.pow2 ? `2^${Math.round(v)} = ${Math.pow(2,Math.round(v)).toLocaleString()}`
                        : p.log ? (+v).toExponential(1)
                        : (p.step<1 ? (+v).toFixed(2) : String(Math.round(v)));
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
    r.oninput=()=>{ knob[p.name]=valOf(r.value); val.textContent=fmt(knob[p.name]);
                    preview(); };
    d.append(lab,r,ends); el.appendChild(d);
  });
}
panel(document.getElementById("knobs_enc"), "encoding and decoder", KNOBS);
panel(document.getElementById("knobs_train"), "training", TRAIN);

let pv=null;
async function preview(){
  clearTimeout(pv);
  pv=setTimeout(async()=>{
    const r=await (await fetch("/api/preview?"+new URLSearchParams(knob))).json();
    if(r.error) return;
    drawLadder(r.ladder, r.info);
    if(!JOBRUNNING) document.getElementById("note").innerHTML=noteHTML(r.info);
  }, 120);
}
let JOBRUNNING=false;

function drawLadder(ladder, info){
  if(!ladder) return;
  let h='<table class="ladder"><tr><th>band</th><th>units</th>'
       +'<th>cycles across</th><th>px / cycle</th></tr>';
  ladder.forEach(L=>{
    h+=`<tr><td><span style="color:${levColor(L.band/LUTMAX)}">&#9632;</span> `
      +`${L.band}</td><td>${L.units}</td>`
      +`<td>${L.f_lo.toFixed(2)} &ndash; ${L.f_hi.toFixed(2)}</td>`
      +`<td>${L.px.toFixed(1)}</td></tr>`; });
  h+="</table>";
  // metrics is {} until the first run, so check the field and not the object.
  if(info && info.n_first!==undefined)
    h+=`<div class="note">${info.n_first.toLocaleString()} first layer + `
      +`${info.n_rest.toLocaleString()} after it</div>`;
  // The comparison that matters here: does the finest band reach the pixels?
  if(!ladder.length){ document.getElementById("ladder").innerHTML=h; return; }
  const fine=ladder[ladder.length-1];
  const px=fine.px;
  h+=`<div class="note">the finest band starts at `
    +`<b>${px.toFixed(1)} px per cycle</b>: `
    + (px > 4 ? `<span style="color:var(--amber)">every wave finer than that has `
               +`to be manufactured by the layers above</span>`
              : `the first layer already reaches the pixels`)
    +`. Equal-count bands, so 16 rows always.</div>`;
  document.getElementById("ladder").innerHTML=h;
}

async function startRun(){
  SEEN_RUNNING=false;
  await fetch("/api/start?"+new URLSearchParams(knob));
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 400); poll();
}
document.getElementById("run").onclick=startRun;
document.getElementById("stop").onclick=()=>fetch("/api/stop");
document.getElementById("clear").onclick=async()=>{ await fetch("/api/clear"); poll(); };
// The live montage: one server-side mode, switched here, redrawn on the next
// refresh of the fit.
const DECMODES=[["adds","what it adds (\u00b1)"],["alone","the band alone"],
                ["upto","bands 0..b"]];
let decmode="adds";
(function(){
  const g=document.getElementById("decmodes");
  DECMODES.forEach(([k,lab])=>{ const b=document.createElement("button");
    b.textContent=lab; b.dataset.k=k;
    b.onclick=async()=>{ decmode=k; paintDec();
      await fetch("/api/decomp_mode?mode="+k); };
    g.appendChild(b); });
  paintDec();
})();
function paintDec(){
  const g=document.getElementById("decmodes");
  [...g.children].forEach(b=>b.setAttribute("aria-pressed", b.dataset.k===decmode));
  const lab=(DECMODES.find(m=>m[0]===decmode)||["",""])[1];
  document.getElementById("cap_mont").textContent=
    `the 16 frequency bands \u2014 ${lab}`;
}
document.getElementById("decompose").onclick=()=>
  window.open("/decompose", "ngp_decompose", "width=1500,height=1000");

// One shared view for every panel: hovering any of them magnifies all of them
// about the same point, so the fit, the error and the level grid can be read
// against each other at the same place rather than eyeballed across panels.
// Panels the magnifier does not touch. The montage is not the picture -- it is
// sixteen small pictures side by side, and magnifying "the same point" across a
// tile grid lands on whichever tile happens to be there.
const NOZOOM=["c_montage"];
function view(cv, iw, ih){
  const s0=Math.min(cv.width/iw, cv.height/ih);
  // With "reference fixed" the first panel stays whole and acts as a navigator;
  // otherwise every panel magnifies together.
  if(!ZOOM.on || NOZOOM.includes(cv.id) || (cv.id==="c_ref" && ZOOM.refFixed))
    return {s:s0, ox:(cv.width-iw*s0)/2, oy:(cv.height-ih*s0)/2, s0};
  const s=s0*ZOOM.f;
  return {s, ox:cv.width/2-ZOOM.u*iw*s, oy:cv.height/2-ZOOM.v*ih*s, s0};
}
function redrawAll(){
  PANELS.forEach(id=>{ if(id==="c_levels") drawLevels(LASTBLOCKS);
                       else if(IMG[id]) blit(document.getElementById(id).getContext("2d"),
                                             document.getElementById(id), IMG[id]); });
  ["c_effmap","c_montage"].forEach(id=>{ if(IMG[id])
    blit(document.getElementById(id).getContext("2d"),
         document.getElementById(id), IMG[id]); });
}
// Every panel can drive the magnifier; with "reference fixed" only the first
// one does, so the whole picture stays available to point at.
PANELS.concat(["c_effmap"]).forEach(id=>{
  const cv=document.getElementById(id);
  cv.addEventListener("mousemove", e=>{
    if(ZOOM.refFixed && id!=="c_ref") return;
    const r=cv.getBoundingClientRect();
    // The canvas is CSS-scaled, so screen pixels are not backing-store pixels.
    const cx=(e.clientX-r.left)/r.width*cv.width, cy=(e.clientY-r.top)/r.height*cv.height;
    const im=IMG[id] || IMG["c_ref"]; if(!im) return;
    const v=view(cv, im.width, im.height);
    ZOOM.u=Math.min(1,Math.max(0,(cx-v.ox)/(im.width*v.s)));
    ZOOM.v=Math.min(1,Math.max(0,(cy-v.oy)/(im.height*v.s)));
    ZOOM.on=true; redrawAll(); zoomNote();
  });
  cv.addEventListener("mouseleave", ()=>{
    if(ZOOM.refFixed && id!=="c_ref") return;
    ZOOM.on=false; redrawAll(); zoomNote(); });
  cv.addEventListener("wheel", e=>{
    e.preventDefault();
    ZOOM.f=Math.min(32, Math.max(1, ZOOM.f*(e.deltaY<0?1.25:0.8)));
    if(ZOOM.f<=1.02){ ZOOM.on=false; } redrawAll(); zoomNote();
  }, {passive:false});
});
function zoomNote(){
  const mode = ZOOM.refFixed
    ? "reference stays whole and marks the region"
    : "all four panels magnify together";
  document.getElementById("zoomnote").textContent = ZOOM.on
    ? `magnifier ${ZOOM.f.toFixed(1)}x at (${(ZOOM.u*100).toFixed(0)}%, `
      +`${(ZOOM.v*100).toFixed(0)}%) — ${mode}; scroll to change, move off to reset`
    : `hover a panel to magnify at that point — ${mode}; scroll to zoom`;
}
function drawImg(id, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(!src){ g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height); return; }
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const v=view(cv, im.width, im.height);
  // Nearest-neighbour once magnified, so the pixels are visible as pixels.
  // Never smooth the montage: its tiles are nearest-neighbour blow-ups of each
  // level's own grid, and smoothing them back is undoing the point.
  g.imageSmoothingEnabled = NOZOOM.includes(cv.id) ? false
    : (!ZOOM.on || (cv.id==="c_ref" && ZOOM.refFixed));
  g.drawImage(im, v.ox, v.oy, im.width*v.s, im.height*v.s);
  if(cv.id==="c_ref" && ZOOM.on && ZOOM.refFixed)
    drawViewport(g, cv, im.width, im.height, v);
}
// The rectangle the other panels are currently showing, drawn on the navigator.
function drawViewport(g, cv, iw, ih, v){
  const z=document.getElementById("c_fit");
  const wImg=z.width/(v.s0*ZOOM.f), hImg=z.height/(v.s0*ZOOM.f);
  const x=v.ox+(ZOOM.u*iw-wImg/2)*v.s, y=v.oy+(ZOOM.v*ih-hImg/2)*v.s;
  g.strokeStyle="#e5484d"; g.lineWidth=1.4;
  g.strokeRect(x, y, wImg*v.s, hImg*v.s);
}

const HCOL=["#4da3ff","#e5a23c","#2ea043","#cf6bd6","#e5484d","#6bd6c9",
            "#9aa4b2","#d6c96b"];
// low bands blue, high bands red
function levColor(t){
  // Bright at both ends: viridis' dark end is invisible on a dark image, and the
  // low bands -- the ones that carry most of a smooth picture -- live there.
  // Blue, turquoise, green, amber, red instead.
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
function drawLevels(bk){
  bk = bk || LASTBLOCKS; LASTBLOCKS = bk;
  const cv=document.getElementById("c_levels"), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!bk || !bk.profile){ document.getElementById("levlegend").textContent=
    "run a fit to see the band profile"; return; }
  const P=bk.profile, pad={l:46,r:12,t:14,b:26};
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const mx=Math.max(...P.map(p=>p.mean))||1;
  const bw=W/P.length;
  g.strokeStyle="#333"; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad.t+H*i/4; g.beginPath();
    g.moveTo(pad.l,y); g.lineTo(pad.l+W,y); g.stroke(); }
  P.forEach((p,i)=>{
    const hgt=H*p.mean/mx;
    g.fillStyle=levColor(p.band/LUTMAX);
    g.fillRect(pad.l+i*bw+1, pad.t+H-hgt, bw-2, hgt);
  });
  // The concentration line, on its own 0-100% scale: how much of a band's total
  // sits in its busiest tenth of the frame. 10% is perfectly spread out.
  g.strokeStyle="#fff"; g.lineWidth=1.4; g.beginPath();
  P.forEach((p,i)=>{ const x=pad.l+(i+0.5)*bw, y=pad.t+H*(1-Math.min(1,p.top10));
    i ? g.lineTo(x,y) : g.moveTo(x,y); });
  g.stroke();
  g.setLineDash([3,3]); g.strokeStyle="#777"; g.beginPath();
  const y10=pad.t+H*0.9; g.moveTo(pad.l,y10); g.lineTo(pad.l+W,y10); g.stroke();
  g.setLineDash([]);
  g.fillStyle="#9a9a9a"; g.font="10px sans-serif"; g.textAlign="right";
  g.fillText(mx.toFixed(3), pad.l-5, pad.t+8);
  g.fillText("0", pad.l-5, pad.t+H);
  g.textAlign="center";
  g.fillText("band", pad.l+W/2, cv.height-8);
  g.fillText("0", pad.l+bw/2, cv.height-16);
  g.fillText(String(P.length-1), pad.l+W-bw/2, cv.height-16);
  g.textAlign="left";
  g.fillStyle="#fff"; g.fillText("concentration", pad.l+4, pad.t+10);
  const spread=P.map(p=>p.top10);
  document.getElementById("levlegend").innerHTML =
    `<div>bars: mean |change in the picture| when that band is released, on the `
    +`band colour scale. White line: what share of that change sits in the `
    +`busiest tenth of the frame &mdash; <b>${(Math.min(...spread)*100).toFixed(0)}`
    +`&ndash;${(Math.max(...spread)*100).toFixed(0)}%</b> here, against 10% for `
    +`perfectly spread out (dashed).</div>`
    +`<div style="margin-top:4px">This panel replaced a "finest band per 64 px `
    +`block" map, which is the question ngp-demo's page asks of a grid. A grid `
    +`level only touches its own cells, so a block can name one; a SIREN band is `
    +`a plane wave across the whole frame, and the map returned band `
    +`${P.length-1} in 255 of 255 blocks. True, and useless.</div>`;
}
function drawCurve(cur, hist){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const W=cv.width, H=cv.height;
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  const all=(hist||[]).map((h,i)=>({pts:h.curve, col:HCOL[i%HCOL.length],
                                    lab:h.label}))
    .concat(cur && cur.length ? [{pts:cur, col:"#ffffff", lab:"current"}] : []);
  if(!all.length) return;
  const pts=all.flatMap(a=>a.pts);
  const tmax=Math.max(...pts.map(p=>p.t), 1e-3);
  const dbs=pts.map(p=>p.psnr).filter(v=>isFinite(v));
  const lo=Math.min(...dbs), hi=Math.max(...dbs);
  const pad={l:52,r:12,t:14,b:34};
  const X=t=>pad.l+Math.log10(1+t)/Math.log10(1+tmax)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-lo)/((hi-lo)||1))*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.fillStyle="#666"; g.font="10px monospace";
  for(let i=0;i<=4;i++){ const v=lo+(hi-lo)*i/4, y=Y(v);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.fillText(v.toFixed(1), 8, y+3); }
  all.forEach(a=>{
    g.strokeStyle=a.col; g.lineWidth=a.col==="#ffffff"?2.2:1.5; g.beginPath();
    a.pts.forEach((p,i)=>{ const x=X(p.t), y=Y(p.psnr);
      i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke(); });
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  g.fillText("training time (s, log)", W/2-56, H-10);
  g.save(); g.translate(14,H/2+34); g.rotate(-Math.PI/2);
  g.fillText("psnr (dB)",0,0); g.restore();
}

function drawHistory(hist){
  if(!hist || !hist.length){ document.getElementById("history").innerHTML=
    '<div class="note">no finished runs yet</div>'; return; }
  let h='<table class="ladder"><tr><th></th><th>settings</th><th>params</th>'
       +'<th>psnr</th><th>s</th></tr>';
  hist.forEach((r,i)=>{
    h+=`<tr><td style="color:${HCOL[i%HCOL.length]}">&#9632;</td>`
      +`<td style="text-align:left">${r.label}</td>`
      +`<td>${r.params.toLocaleString()}</td><td>${r.psnr.toFixed(2)}</td>`
      +`<td>${r.seconds.toFixed(1)}</td></tr>`; });
  document.getElementById("history").innerHTML=h+"</table>";
}

async function poll(){
  const r=await (await fetch("/api/state")).json();
  JOBRUNNING=r.running;
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";
  if(r.metrics && r.metrics.n_total!==undefined)
    document.getElementById("note").innerHTML=noteHTML(r.metrics);
  else if(r.note) document.getElementById("note").textContent=r.note;
  if(r.stamp!==LAST){
    LAST=r.stamp;
    drawImg("c_ref", r.images.reference); drawImg("c_fit", r.images.fit);
    drawImg("c_err", r.images.error);   drawImg("c_effmap", r.images.levels);
    drawImg("c_montage", r.images.montage);
    drawLevels(r.blocks);
    drawCurve(r.curve, r.history); drawLadder(r.ladder, r.metrics);
    drawHistory(r.history);
    const m=r.metrics||{};
    document.getElementById("stats").innerHTML = m.psnr===undefined ? "press run"
      : `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `
       +`${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; psnr <b>${m.psnr.toFixed(2)}</b> dB`
       +`<br><b>${m.n_total.toLocaleString()}</b> parameters `
       +`(<b>${(m.fraction_of_values*100).toFixed(1)}%</b> of the `
       +`${m.n_values.toLocaleString()} reference RGB values) &nbsp;&middot;&nbsp; `
       +`finest first-layer cycle covers `
       +`<b>${m.finest_px_per_cycle.toFixed(1)}</b> px`
       +` &nbsp;&middot;&nbsp; &omega;<sub>0</sub> ${m.omega.toFixed(1)}`;
  }
  // Only stop once the run has actually been observed running: a not-yet-started
  // job and a finished one look identical from here.
  if(r.running) SEEN_RUNNING=true;
  if(SEEN_RUNNING && !r.running && POLL){ clearInterval(POLL); POLL=null; }
}
zoomNote(); preview(); poll();
// Open on a running fit rather than an empty page: the default configuration is
// the one worth seeing first, and stopping it costs one click.
startRun();
</script></body></html>
"""


DECOMP_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>one band at a time</title>
<style>__CSS__
.mont{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.mont .cell{background:#141414;border:1px solid #262626;padding:6px;text-align:center}
.mont img{width:100%;display:block;background:#000}
.mont .lab{font-size:11px;color:#9a9a9a;margin-top:4px}
.mont .hash{color:#4da3ff}
</style></head><body><div class="wrap">
<h1>one band at a time</h1>
<p class="sub">The first layer is <b>a bank of plane waves</b>, one per unit, at
&omega;<sub>0</sub>|w<sub>i</sub>|/2&pi; cycles across the picture. Sorted by that and
cut into sixteen equal-count bands, each panel below is the fit rendered with <b>one
band of units left switched on</b> and the rest gated to zero &mdash; same weights, same
layers, a sixteenth of the first layer's output. <b>They do not add up:</b> every
layer after the first mixes what it is given, so a band alone is not a term of a
sum. The <i>what the band adds</i> view is the honest decomposition &mdash; it
differences two renders instead of isolating one band, and its first tile is the
<b>baseline</b> those differences start from. That baseline is not black: with every
unit gated off the first layer emits zeros and the layers above return a constant.
Baseline plus the fifteen differences is the fit, exactly.</p>
<div class="controls"><div class="group"><div class="label">view</div>
  <div class="seg" id="modes"></div></div>
<div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="refresh">refresh from the current fit</button></div>
</div></div>
<div class="note" id="note">rendering&hellip;</div>
<div class="mont" id="mont"></div>
</div><script>
const MODES=[["adds","what the band adds"],["alone","the band alone"],
             ["upto","bands 0..b"]];
let mode="adds";
const seg=document.getElementById("modes");
MODES.forEach(([k,lab])=>{ const b=document.createElement("button");
  b.textContent=lab; b.onclick=()=>{ mode=k; draw(); paint(); }; b.dataset.k=k;
  seg.appendChild(b); });
function paint(){ [...seg.children].forEach(b=>
  b.setAttribute("aria-pressed", b.dataset.k===mode)); }
async function draw(){
  document.getElementById("note").textContent="rendering...";
  const r=await (await fetch("/api/decompose?mode="+mode)).json();
  if(r.error){ document.getElementById("note").textContent=r.error;
               document.getElementById("mont").innerHTML=""; return; }
  document.getElementById("note").innerHTML=
    `${r.panels.length} tiles over ${r.n_levels} bands, rendered at `
   +`${r.render}` + (mode==="adds"
      ? ` &mdash; band b minus band b-1, <span style="color:#4a8bff">blue`
       +`</span> negative, black zero, <span style="color:#ff4b47">red</span> `
       +`positive, fixed scale &plusmn;__DMAX__`
      : ``);
  document.getElementById("mont").innerHTML = r.panels.map(p=>
    `<div class="cell"><img src="${p.png}">`
   +`<div class="lab">${p.label} &middot; ${p.cells} units &middot; `
   +`sampled ${p.grid} &middot; `
   +`${p.px_per_cell.toFixed(1)} px/cycle`
   +` &middot; ${p.kind==="base" ? "the baseline, mean" :
                  (mode==="adds"?"mean |&Delta;|":"std")} `
   +`${p.amp.toFixed(4)}</div></div>`
  ).join("");
}
paint(); draw();
document.getElementById("refresh").onclick=draw;
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
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
            page = (PAGE.replace("__CSS__", CSS)
                        .replace("__ABOUT__", ABOUT_HTML)
                        .replace("__HELP__", INTERFACE_IMAGE)
                        .replace("__KNOBS__", json.dumps(KNOBS))
                        .replace("__TRAIN__", json.dumps(TRAIN_KNOBS))
                        .replace("__DEF__", json.dumps(DEFAULTS))
                        .replace("__LUTMAX__", str(LEVEL_LUT_MAX))
                        .replace("__ERRMAX__", f"{ERROR_LUT_MAX:g}"))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/decompose":
            return self._send(DECOMP_PAGE.replace("__CSS__", CSS)
                                         .replace("__DMAX__", f"{DECOMP_LUT_MAX:g}"),
                              "text/html; charset=utf-8")
        if u.path == "/api/decomp_mode":
            m = q.get("mode", "adds")
            if m in ("alone", "adds", "upto"):
                DECOMP["mode"] = m
            return self._send(json.dumps({"mode": DECOMP["mode"]}),
                              "application/json")
        if u.path == "/api/decompose":
            with LOCK:
                model, shape = MODEL["model"], MODEL["shape"]
            if model is None:
                return self._send(json.dumps({"error": "run a fit first"}),
                                  "application/json")
            try:
                # A COPY: unit_gain lives on the model the trainer may still be
                # stepping, and a decomposition that gated units underneath a
                # running fit would corrupt it.
                import copy
                snap = copy.deepcopy(model)
                out = decompose(snap, shape, self.device, q.get("mode", "adds"))
                for p in out["panels"]:
                    p["png"] = png_data_uri(p.pop("rgb"))
                print(f"[decompose] {len(out['panels'])} tiles, mode "
                      f"{out['mode']}, rendered at {out['render']}", flush=True)
                return self._send(json.dumps(out), "application/json")
            except Exception as e:
                print(f"[decompose] failed: {type(e).__name__}: {e}", flush=True)
                return self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                  "application/json")
        if u.path == "/api/preview":
            p = _params(q)
            try:
                shape = image_shape(DEFAULT_IMAGE, max(1, int(p["downsample"])))
                model, ladder = build(p, shape)
                info = describe(model, shape)
                note = (f"{info['width']}x{info['height']}, {info['n_total']:,} "
                        f"parameters ({info['fraction_of_values']*100:.1f}% of the "
                        f"{info['n_values']:,} reference values), omega_0 "
                        f"{info['omega']:g}, {info['f_min']:.2f}..{info['f_max']:.2f} "
                        f"cycles, {info['finest_px_per_cycle']:.1f} px per finest "
                        f"cycle")
                return self._send(json.dumps({"ladder": ladder, "info": info,
                                              "note": note}), "application/json")
            except Exception as e:
                return self._send(json.dumps({"error": str(e)}), "application/json")
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
            threading.Thread(target=train_job, args=(_params(q), self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clienterror":
            print(f"[client] {q.get('msg', '')}", flush=True)
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            if JOB["running"]:
                print("[stop] requested", flush=True)
            STOP.set()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clear":
            with LOCK:
                JOB["history"] = []
                JOB["stamp"] += 1
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/state":
            with LOCK:
                return self._send(json.dumps(JOB), "application/json")
        self.send_error(404)


def _params(q):
    p = dict(DEFAULTS)
    for k, v in q.items():
        p[k] = float(v) if _isnum(v) else v
    return p


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    global DEFAULT_IMAGE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    DEFAULT_IMAGE = a.image
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
