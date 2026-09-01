#!/usr/bin/env python
"""Fit a scalar field f(x, y, t) from a zarr with a SIREN, in a browser.

    python scripts/gui_scalar_time.py                     # http://localhost:8125
    python scripts/gui_scalar_time.py --zarr-glob "/path/to/*/field.zarr"

Written for `Plexus/prototype/graphcast/log/toy2d*/field.zarr`, which is two PDEs
laid on top of each other, and that is exactly why it is worth fitting: the two
components want opposite things from one representation.

Measured on that store before any of this was written:

  coarse `u`, 256^2      0.0% of its energy above 32 cycles across the frame;
                         lag-1 autocorrelation 0.998, and still correlated with
                         frame 0 at 0.5 eighteen frames later.
  fine `v`, 1024^2       73.6% of its energy ABOVE 32 cycles, 15 px per cycle
                         inside a disc, on 15.4% of the pixels; lag-1
                         autocorrelation 0.829, and past 0.5 after ONE frame.

The hash-grid twin of this page answers that with two settings -- 4 px per finest
cell, and a time axis AT the frame spacing.  A SIREN has neither: x, y and t are
three inputs to one bank of plane waves, and omega_0 is the only scale it has.
So this page asks whether one omega_0 can hold a field whose two halves are two
decades apart in space and an order of magnitude apart in time, and the `field`
selector fits either half alone to see which one it gives up.
"""

from __future__ import annotations

import argparse
import base64
import glob
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
from siren.utils import pixel_centers, render
from PIL import Image, ImageDraw

from siren.webui import (ABOUT_HTML, CSS, cmap_png, display_range,
                        field_png, png_data_uri, signed_rgb)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The generator writes one store per run and names the directory after it, so a
# hard-coded path goes stale the next time it is regenerated -- it already did
# once while this was written.  Each store carries a summary.json saying which
# part it holds, which is what the dataset toggle is built from rather than the
# directory name.
# TWO PLACES, both by default: the toy generator's own output, and the real
# datasets scripts/prepare_datasets.py writes into this repo. A page that only
# looked at the first showed a menu of toys while the real cuts sat on disk
# beside it, which is exactly what happened.
ZARR_GLOB = ["/workspace/Plexus/prototype/graphcast/log/*/field.zarr",
             os.path.join(ROOT, "data", "*", "field.zarr"),
             # and the twin's, because scripts/prepare_datasets.py lives there
             # and writes 154 MB that neither repo should hold twice
             os.path.join(os.path.dirname(ROOT), "ngp-demo", "data", "*",
                          "field.zarr")]


def _grid_size(store, k):
    """Cells along x for that field, from metadata alone: the label's suffix."""
    meta = os.path.join(store, k, "grid", ".zarray")
    try:
        with open(meta) as f:
            return json.load(f)["shape"][-1]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _grid_rank(store):
    """Length of the first grid array's shape, from metadata alone -- no read."""
    for k in ("u", "v"):
        meta = os.path.join(store, k, "grid", ".zarray")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    return len(json.load(f)["shape"])
            except (OSError, ValueError, KeyError):
                return None
    return None


def datasets(pattern=None):
    """{label: (u_store or None, v_store or None)} for whichever runs are on disk.

    A store holds `u` (the slow wave), `v` (the Kuramoto discs), or both, and its
    summary.json says which.  The menu offers each part on its own, and one SUM
    PER COARSE RESOLUTION -- "u 64 + v", "u 256 + v" -- because the generator
    writes the coarse field at more than one and the whole question this page
    asks is what the encoder does when the two scales are far apart.

    A sum is paired within one run when it can be: the both-run's own u and v.
    Otherwise it pairs a standalone coarse run with the v of the both-run, which
    is legitimate here and measured to be so -- the two v arrays are bit
    identical (max |difference| 0.0), the fine field being deterministic given
    the seed.  The coarse trajectories are NOT shared that way, so a mixed pair
    is a different realisation of the same process and the page says so.
    """
    us, vs, both, out = {}, {}, {}, {}
    pats = pattern or ZARR_GLOB
    pats = [pats] if isinstance(pats, str) else list(pats)
    found = sorted({d for pat in pats for d in glob.glob(pat)},
                   key=os.path.getmtime)
    for d in found:
        part, name = None, os.path.basename(os.path.dirname(d))
        # A store being written right now has a directory and no .zgroup yet.
        # Skipping it beats crashing on it: the generator is often running.
        if not os.path.exists(os.path.join(d, ".zgroup")):
            continue
        # And this page is f(x, y, t).  The same generator writes toy3d runs
        # whose grids are (T, 1, Z, Y, X); they are not this page's problem, and
        # picking one up silently would put a 3D volume behind a 2D selector.
        rank = _grid_rank(d)
        if rank is not None and rank != 4:
            continue
        try:
            with open(os.path.join(os.path.dirname(d), "summary.json")) as f:
                part = json.load(f).get("part")
        except (OSError, ValueError):
            pass
        if part is None:                       # no summary: fall back on the name
            part = ("coarse" if "coarse" in name else
                    "fine" if "fine" in name else "both")
        # The toy generator writes one part per run and the menu names them by
        # part and resolution.  Anything else -- a real dataset prepared by
        # scripts/prepare_datasets.py -- is named after its own directory,
        # because "fine v 256" would collapse two zapbench cuts of the same size
        # into one entry and silently drop a dataset.
        if not name.startswith("toy"):
            out[name.replace("_", " ")] = (None, d)
            continue
        if part == "both":
            both[_grid_size(d, "u")] = d
            us.setdefault(_grid_size(d, "u"), d)
            vs.setdefault(_grid_size(d, "v"), d)
        elif part == "coarse":
            us[_grid_size(d, "u")] = d
        else:
            vs[_grid_size(d, "v")] = d

    v_src = (next(iter(both.values()), None)
             or (sorted(vs.items())[-1][1] if vs else None))
    for n, d in sorted(us.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if v_src is not None:
            # same-run pairing wherever the both-run's own u is this size
            out[f"u {n} + v"] = (both.get(n, d), both.get(n, v_src))
        out[f"coarse u {n}"] = (d, None)
    for n, d in sorted(vs.items(), key=lambda kv: (kv[0] is None, kv[0])):
        out[f"fine v {n}"] = (None, d)
    return out


DATASETS = datasets()
DEFAULT_ZARR = DATASETS.get("sum") or next(iter(DATASETS.values()), "")

LEVEL_LUT_MAX = 15                   # 16 bands, always
ERROR_LUT_MAX = 0.20                 # signed, fixed: the field itself is +-1

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "note": "", "stamp": 0, "frames": [],
       "per_frame": [], "levels": {}, "levels_live": 0.0}
LOCK = threading.Lock()
STOP = threading.Event()
DATA = {}
# The finished fit, kept so the play button can render any frame on demand.
PLAY: dict = {}


# ----------------------------------------------------------------- the data


def load_field(which, down, device, stores=None):
    """(T, H, W) on the device, the frame count, the spatial size, the source.

    NEAREST, not bilinear, when the coarse field is carried up to the fine grid:
    it is discrete at its own resolution and smoothing it would hand the fit
    structure the PDE never produced.  One coarse cell is one block of fine
    cells, which is what the generator writes as well.

    When a sum pairs two runs, their frame counts differ -- the coarse runs are
    written at 401 frames against the fine 201 -- so the longer one is strided
    to meet the shorter.  Both cover the same span; only the sampling differs.
    """
    stores = stores or DATASETS
    pair = stores.get(which)
    if not pair:
        raise FileNotFoundError(
            f"no {which} dataset under {ZARR_GLOB} -- found "
            f"{sorted(stores) or 'nothing'}")
    u_store, v_store = pair
    key = (u_store, v_store, down, str(device))
    if key in DATA:
        return DATA[key]
    import zarr

    def grid(store, k):
        g = zarr.open(store, mode="r")
        if k not in set(g.group_keys()):
            raise KeyError(f"{os.path.basename(os.path.dirname(store))} has no "
                           f"{k}/grid")
        return torch.from_numpy(np.asarray(g[f"{k}/grid"][:, 0])).to(device)

    # STREAMED IN BLOCKS OF FRAMES, not read whole. The fine field is
    # 201 x 1024^2 x 4 B = 843 MB at full resolution, and this page shares a GPU
    # -- the first version of this asked for all of it at once and died with
    # 632 MB free. Pooling each block as it lands means the peak is one block,
    # and at downsample 2 the result is a quarter of the size.
    def stream(chunk=16):
        import zarr
        gu = zarr.open(u_store, mode="r") if u_store else None
        gv = zarr.open(v_store, mode="r") if v_store else None
        nu = gu["u/grid"].shape[0] if gu is not None else None
        nv = gv["v/grid"].shape[0] if gv is not None else None
        step = 1
        if nu and nv and nu != nv:              # 401 coarse frames against 201
            step = max(1, round((nu - 1) / max(1, nv - 1)))
        n = nv if nv is not None else nu
        out = []
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            if gv is None:
                a = torch.from_numpy(
                    np.asarray(gu["u/grid"][i0 * step:i1 * step:step, 0])).to(device)
            elif gu is None:
                a = torch.from_numpy(np.asarray(gv["v/grid"][i0:i1, 0])).to(device)
            else:
                v = torch.from_numpy(np.asarray(gv["v/grid"][i0:i1, 0])).to(device)
                u = torch.from_numpy(
                    np.asarray(gu["u/grid"][i0 * step:i1 * step:step, 0])).to(device)
                u = u[:v.shape[0]]
                a = F.interpolate(u[:, None], size=v.shape[-2:],
                                  mode="nearest")[:, 0] + v[:u.shape[0]]
                del u, v
            if down > 1:
                a = F.avg_pool2d(a[:, None], down)[:, 0]
            out.append(a)
        return torch.cat(out, 0)

    a = stream()
    src = " + ".join(sorted({os.path.basename(os.path.dirname(x))
                             for x in (u_store, v_store) if x}))
    DATA[key] = (a.contiguous(), a.shape[0], a.shape[1], a.shape[2], src)
    return DATA[key]


@torch.no_grad()
def sample_field(vol, xyt):
    """Trilinear lookup into (T, H, W) at xyt in [0,1]^3 -> (N,).

    Minus half a voxel on every axis, for the reason ngp/utils.BilinearImage now
    carries: sample j covers [j/n, (j+1)/n) and sits at its centre, and without
    the shift a query at a sample's own centre returns the average of it and its
    neighbour -- a half-voxel blur of the whole target.
    """
    T, H, W = vol.shape
    size = torch.tensor([W, H, T], device=xyt.device, dtype=xyt.dtype)
    p = (xyt * size - 0.5).clamp(min=torch.zeros(3, device=xyt.device),
                                 max=size - 1)
    i0 = torch.floor(p)
    f = (p - i0).unsqueeze(-1)
    i0 = i0.long()
    x0, y0, t0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    t1 = (t0 + 1).clamp(max=T - 1)
    fx, fy, ft = f[:, 0, 0], f[:, 1, 0], f[:, 2, 0]
    c00 = vol[t0, y0, x0] * (1 - fx) + vol[t0, y0, x1] * fx
    c01 = vol[t0, y1, x0] * (1 - fx) + vol[t0, y1, x1] * fx
    c10 = vol[t1, y0, x0] * (1 - fx) + vol[t1, y0, x1] * fx
    c11 = vol[t1, y1, x0] * (1 - fx) + vol[t1, y1, x1] * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - ft) + c1 * ft


def frame_coords(h, w, t, device):
    xy = pixel_centers(h, w, device)
    return torch.cat([xy, torch.full((xy.shape[0], 1), float(t), device=device)], 1)


# ------------------------------------------------------------------- the fit


N_BANDS = 16


def build(p, w, h, n_frames):
    """The SIREN, with the time axis stretched to the requested scale.

    There is no time axis to cap here: x, y and t are three inputs to one bank
    of plane waves and omega_0 multiplies all of them.  The equivalent knob is
    the coordinate itself -- feeding t * s makes every first-layer wave s times
    faster along t -- so "frames per finest cycle" is met by solving for s from
    the initialised weights and reading the result back.
    """
    m = SirenField(
        n_input_dims=3, n_output_dims=1, output_activation="none",
        width=int(p["width"]), hidden_layers=int(p["hidden_layers"]),
        omega_0=float(p["omega_0"]), outermost_linear=True)
    fpc = max(1.0, float(p["frames_per_finest_cycle"]))
    want = n_frames / fpc                      # cycles across the run
    have = float(m.frequencies(axis=2).max())  # before any stretch
    if have > 1e-9:
        m.input_scale[2] = want / have
    return m


@torch.no_grad()
def band_map_at(model, h, w, t, device, sub=4):
    """Per pixel, the effective frequency band at time t.

    Not "the finest band per block", which is what the hash-grid twin draws: a
    grid level only touches its own cells, so a block can name the one that moved
    it, while a SIREN band is a plane wave over the whole frame and the block
    version names the top band nearly everywhere.  The amplitude-weighted mean
    band does vary, and on this field it has something to say -- the fine
    component sits on 15% of the pixels, so the discs should pull it up.
    """
    hs, ws = max(8, h // sub), max(8, w // sub)
    xyt = frame_coords(hs, ws, t, device)
    prev, deltas = None, []
    for k in range(N_BANDS + 1):
        model.set_band_window(float(k), N_BANDS)
        out = model(xyt).reshape(hs, ws)
        if prev is not None:
            deltas.append((out - prev).abs())
        prev = out
    model.set_band_window(float(N_BANDS), N_BANDS)
    D = torch.stack(deltas)
    b = torch.arange(D.shape[0], device=D.device, dtype=D.dtype)
    eff = (D * b[:, None, None]).sum(0) / D.sum(0).clamp(min=1e-9)
    return {"png": cmap_png(eff.cpu().numpy(), N_BANDS - 1, "levels"),
            "mean": float(eff.mean()), "lo": float(eff.min()),
            "hi": float(eff.max()), "n_levels": N_BANDS}


# What a band adds, signed. NOT a constant: these datasets run from a +-1 wave to
# raw counts in the hundreds, and a fixed +-0.1 saturates every tile of the
# second into one flat red. Set per run from the shown range, like the error
# panel, and reported in the caption.
MONTAGE_LUT_FRAC = 0.05


@torch.no_grad()
def band_montage(model, h, w, t, device, vmax, side=110, cols=4):
    """The 16 frequency bands at one time slice, as one 4x4 picture.

    The twin of the hash grid's level montage, and the same question: what does
    each scale ADD.  Two differences, both measured rather than assumed.  Every
    tile is rendered at the TILE resolution and not at its band's own Nyquist,
    because the layers above the first manufacture harmonics -- 63-93% of a
    tile's spectral energy sits above its band's own top frequency.  And the
    first tile is the baseline the differences start from, which is not black:
    with every unit gated off the first layer emits zeros and the rest of the
    network returns a constant.
    """
    sub = max(1, int(round(max(h, w) / side)))
    hs, ws = max(8, h // sub), max(8, w // sub)
    c = frame_coords(hs, ws, t, device)
    keep = model.unit_gain.clone()
    cache = {}

    def upto(k):
        if k not in cache:
            g = torch.zeros_like(model.unit_gain)
            if k >= 0:
                g[model.band_of(N_BANDS) <= k] = 1.0
            model.set_unit_gain(g)
            cache[k] = render(model, c, (hs, ws))
        return cache[k]

    bands = list(range(N_BANDS))[-(cols * cols - 1):]
    tiles, labels, raw = [], [], []
    try:
        b0 = bands[0] - 1
        a = upto(b0)
        rgb = (np.clip(a.cpu().numpy() / max(1e-6, float(a.abs().max())), -1, 1)
               * 0.5 + 0.5)
        tiles.append((matplotlib.colormaps["viridis"](rgb)[..., :3] * 255).astype(np.uint8))
        labels.append(f"0..{b0}")
        for b in bands:
            d = upto(b) - upto(b - 1)
            tiles.append(signed_rgb(d.cpu().numpy(), vmax))
            labels.append(f"B{b}")
    finally:
        model.set_unit_gain(keep)

    return _sheet(*_scale(tiles, labels, raw, vmax, size=(hs, ws)), cols)


@torch.no_grad()
def band_kymograph(model, h, w, n_frames, device, vmax, side=110, cols=4):
    """The temporal pendant of the band montage: what each band adds, in TIME.

    One horizontal line through the field swept over every frame, so each tile
    is x across and t down.  A SIREN has nothing to label a tile with here: its
    bands are ordered by |w_i|, which mixes the x, y AND t components of the
    same weight vector, so a band has no separate time resolution to quote --
    which is itself the difference from the grid twin, where every level carries
    its own cells-along-t.
    """
    ny = max(16, min(side, n_frames))
    nx = max(16, side)
    y = torch.full((nx * ny,), 0.5, device=device)
    xs = pixel_centers(1, nx, device)[:, 0].repeat(ny)
    ts = torch.arange(ny, device=device).float().repeat_interleave(nx) / max(1, ny - 1)
    xyt = torch.stack([xs, y, ts], dim=1)
    keep = model.unit_gain.clone()
    cache = {}

    def upto(k):
        if k not in cache:
            g = torch.zeros_like(model.unit_gain)
            if k >= 0:
                g[model.band_of(N_BANDS) <= k] = 1.0
            model.set_unit_gain(g)
            cache[k] = model(xyt)[:, 0].reshape(ny, nx)
        return cache[k]

    bands = list(range(N_BANDS))[-(cols * cols - 1):]
    tiles, labels, raw = [], [], []
    try:
        b0 = bands[0] - 1
        a = upto(b0)
        v = np.clip(a.cpu().numpy() / max(1e-6, float(a.abs().max())), -1, 1) * .5 + .5
        tiles.append((matplotlib.colormaps["viridis"](v)[..., :3] * 255).astype(np.uint8))
        labels.append(f"0..{b0}")
        for b in bands:
            d = upto(b) - upto(b - 1)
            tiles.append(signed_rgb(d.cpu().numpy(), vmax))
            labels.append(f"B{b}")
    finally:
        model.set_unit_gain(keep)
    return _sheet(*_scale(tiles, labels, raw, vmax), cols)


def _scale(tiles, labels, raw, hint, size=None):
    """Colour the difference tiles on a scale taken from the differences.

    A fraction of the FIELD's range is the wrong scale for them: on the zapbench
    plane the field runs to 1235 counts and a level's contribution to hundreds,
    so 5% of the field saturated every tile into one flat red.  The 99th
    percentile of |difference| over the whole sheet puts the scale where the
    differences actually are, and it is written onto the first tile so the sheet
    carries its own units.
    """
    if raw:
        allv = np.concatenate([d.ravel() for _, d in raw])
        vmax = float(np.percentile(np.abs(allv), 99)) or abs(hint) or 1e-3
        for i, d in raw:
            rgb = signed_rgb(d, vmax)
            if size is not None and rgb.shape[:2] != tuple(size):
                rgb = np.asarray(Image.fromarray(rgb).resize(
                    (size[1], size[0]), Image.NEAREST))
            tiles[i] = rgb
        labels[0] = f"{labels[0]}  +-{vmax:.3g}"
    return tiles, labels


def _sheet(tiles, labels, cols=4, gap=2):
    """Tiles into one labelled picture."""
    th, tw, _ = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * (th + gap) - gap, cols * (tw + gap) - gap, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // cols) * (th + gap), (i % cols) * (tw + gap)
        sheet[y:y + th, x:x + tw] = tile
    im = Image.fromarray(sheet)
    dr = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        dr.text(((i % cols) * (tw + gap) + 2, (i // cols) * (th + gap) + 1), lab,
                fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()



def _frame_pngs(i):
    """target and fit for frame i, from the cache when it is there.

    The renders are cheap -- 4 ms for the model, 1.7 ms for each png at 256x166
    -- but at panel resolution over a round trip they came to about 40 ms, which
    is the whole of the playback budget.  Every frame is rendered once, kept as
    the two data URIs the page will ask for, and served from memory after that:
    256 frames of two panels is about 50 MB and playback stops touching the GPU.
    """
    hit = PLAY["cache"].get(i)
    if hit is not None:
        return hit
    t = i / max(1, PLAY["n_frames"] - 1)
    with torch.no_grad():
        c = frame_coords(PLAY["h"], PLAY["w"], t, PLAY["device"])
        fit = render(PLAY["model"], c, (PLAY["h"], PLAY["w"]))
    # The residual travels with them: a fit that looks right beside its target
    # and a fit whose error is structured look identical until they are put side
    # by side, and playback is exactly where that shows.
    d = (fit - PLAY["vol"][i]).cpu().numpy()
    out = (field_png(PLAY["vol"][i].cpu().numpy(), PLAY["vmax"],
                     lo=PLAY["lo"], hi=PLAY["hi"]),
           field_png(fit.cpu().numpy(), PLAY["vmax"], lo=PLAY["lo"], hi=PLAY["hi"]),
           png_data_uri(signed_rgb(d, PLAY["err_max"])))
    PLAY["cache"][i] = out
    return out


def _prefetch(token, budget=1024):
    """Fill the frame cache in the background as soon as a fit finishes.

    On its own thread and abandoned the moment another run starts, so a fit is
    never waiting on it.  Skipped above `budget` frames, where the cache would
    cost more memory than the playback is worth.
    """
    n = PLAY.get("n_frames", 0)
    if n > budget:
        return
    for i in range(n):
        if PLAY.get("token") != token:
            return
        try:
            _frame_pngs(i)
        except Exception:
            return
        PLAY["ready"] = i + 1
    print(f"[play  ] {n} frames cached, playback is now local", flush=True)

def train_job(p, device):
    try:
        down = max(1, int(p["downsample"]))
        vol, n_frames, h, w, store = load_field(p["field"], down, device)
        vmax = float(vol.abs().max())
        # The colour range comes from the DATA, not from its extreme: a redox
        # ratio runs to 2.88 with a 99.9th percentile of 1.24, and a symmetric
        # ramp on the extreme spends itself on values that never occur. The
        # error scale follows it -- 5% of the displayed span -- rather than
        # being a constant that suits one dataset.
        lo, hi = display_range(vol)
        err_max = max(1e-6, 0.05 * (hi - lo))
        PLAY.clear()
        PLAY["token"] = PLAY.get("token", 0) + 1
        torch.manual_seed(0)
        model = build(p, w, h, n_frames).to(device)
        n_enc, n_mlp = model.n_parameters()
        freq = model.frequencies()
        # Held-out frames: every k-th one is never sampled, and the score on
        # them is what says whether the fit interpolates in time or memorises.
        hold = int(p["holdout"])
        held = set(range(1, n_frames - 1, hold)) if hold > 1 else set()
        train_t = np.array([i for i in range(n_frames) if i not in held],
                           dtype=np.float32) / max(1, n_frames - 1)
        train_t = torch.from_numpy(train_t).to(device)
        n_values = n_frames * h * w
        compression = n_values / max(1, n_enc + n_mlp)
        picks = [0, n_frames // 2, n_frames - 1]
        note = (f"{p['field']} dataset "
                f"({os.path.basename(os.path.dirname(store))}), "
                f"{n_frames} frames of {w}x{h}, "
                f"shown {lo:.2f}..{hi:.2f} of {float(vol.min()):.2f}..{float(vol.max()):.2f}, error +-{err_max:.3f}; width {int(p['width'])} x "
                f"{int(p['hidden_layers'])} layers, omega_0 "
                f"{float(p['omega_0']):g}, first-layer waves "
                f"{float(freq.min()):.2f}..{float(freq.max()):.2f} cycles "
                f"= {w / max(float(freq.max()), 1e-6):.0f} px per finest cycle; "
                f"t stretched x{float(model.input_scale[2]):.1f} for "
                f"{n_frames / max(float(model.frequencies(axis=2).max()), 1e-9):.1f} "
                f"frames per finest cycle along t"
                + (f"; {len(held)} frames held out" if held else ""))
        print(f"[run] {note}", flush=True)
        print(f"[params] {n_enc + n_mlp:,} total = {n_enc:,} in the first layer "
              f"({int(p['width'])} plane waves over x, y and t) + {n_mlp:,} "
              f"after it", flush=True)
        print(f"[size  ] {n_enc + n_mlp:,} parameters against {n_values:,} stored "
              f"values = {compression:.1f}x compression "
              f"({100 / compression:.1f}% of the field)", flush=True)
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], per_frame=[], levels={}, note=note, frames=picks,
                       metrics={"n_parameters": n_enc + n_mlp, "n_table": n_enc,
                                "n_values": n_values, "compression": compression,
                                "n_decoder": n_mlp, "n_levels_total": N_BANDS,
                                "n_frames": n_frames, "vmax": vmax,
                                "n_held": len(held)},
                       images={f"target{i}": field_png(
                           vol[t].cpu().numpy(), vmax, lo=lo, hi=hi)
                           for i, t in enumerate(picks)},
                       stamp=JOB["stamp"] + 1)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps, batch = int(p["steps"]), int(p["batch"])
        every = max(1, steps // 30)
        t0 = time.perf_counter()
        for step in range(steps + 1):
            if STOP.is_set():
                break
            if int(p["coarse_to_fine"]):
                a = 4 + (N_BANDS - 4) * min(1.0, step / max(1, steps * 0.5))
                model.set_band_window(a, N_BANDS)
                JOB["levels_live"] = round(float(a), 2)
            else:
                JOB["levels_live"] = float(N_BANDS)
            xy = torch.rand(batch, 2, device=device)
            ti = train_t[torch.randint(len(train_t), (batch,), device=device)]
            xyt = torch.cat([xy, ti[:, None]], 1)
            pred = model(xyt)[:, 0]
            loss = ((pred - sample_field(vol, xyt)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % every == 0 or step == steps:
                with torch.no_grad():
                    imgs, errs, lv = {}, [], {}
                    for i, fr in enumerate(picks):
                        c = frame_coords(h, w, fr / max(1, n_frames - 1), device)
                        fit = render(model, c, (h, w))
                        d = (fit - vol[fr])
                        imgs[f"fit{i}"] = field_png(fit.cpu().numpy(), vmax,
                                                    lo=lo, hi=hi)
                        imgs[f"err{i}"] = png_data_uri(
                            signed_rgb(d.cpu().numpy(), err_max))
                        errs.append(float(d.pow(2).mean()))
                        lv[str(i)] = band_map_at(model, h, w,
                                                 fr / max(1, n_frames - 1), device)
                    imgs["montage"] = band_montage(
                        model, h, w, picks[1] / max(1, n_frames - 1), device,
                        err_max)
                    imgs["kymo"] = band_kymograph(model, h, w, n_frames,
                                                   device, err_max)
                    # every frame, on a coarse grid: cheap, and the only way to
                    # see a fit that is good at the ends and lost in between
                    hs, ws = h // 4, w // 4
                    per = []
                    for fr in range(0, n_frames, max(1, n_frames // 60)):
                        c = frame_coords(hs, ws, fr / max(1, n_frames - 1), device)
                        f2 = render(model, c, (hs, ws))
                        ref = F.avg_pool2d(vol[fr][None, None], 4)[0, 0]
                        mse = float((f2 - ref).pow(2).mean())
                        per.append({"t": fr, "psnr": psnr_db(mse, vmax),
                                    "held": fr in held})
                mse = float(np.mean(errs))
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = time.perf_counter() - t0
                    JOB["curve"].append({"step": step, "psnr": psnr_db(mse, vmax),
                                         "loss": loss.detach().item()})
                    JOB["per_frame"] = per
                    JOB["metrics"] = {**JOB["metrics"], "psnr": psnr_db(mse, vmax),
                                      "psnr_held": (float(np.mean(
                                          [10 ** (-q["psnr"] / 10) for q in per
                                           if q["held"]])) if held else None)}
                    if held:
                        JOB["metrics"]["psnr_held"] = -10 * math.log10(
                            max(1e-12, JOB["metrics"]["psnr_held"]))
                    JOB["images"].update(imgs)
                    JOB["levels"] = lv
                    JOB["stamp"] += 1
        PLAY.update(model=model, vol=vol, n_frames=n_frames, vmax=vmax,
                    lo=lo, hi=hi, err_max=err_max, h=h, w=w, device=device,
                    cache={}, ready=0)
        threading.Thread(target=_prefetch, args=(PLAY.get("token"),),
                         daemon=True).start()
    except Exception as e:
        print(f"[run] failed: {type(e).__name__}: {e}", flush=True)
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1
            m = JOB["metrics"]
        print(f"[done ] {JOB['seconds']:.1f}s  psnr {m.get('psnr', 0):.2f} dB"
              + (f"  held-out {m['psnr_held']:.2f} dB" if m.get("psnr_held")
                 else ""), flush=True)


def psnr_db(mse, vmax):
    """Peak signal to noise against the field's own peak-to-peak (2*vmax)."""
    return float(10 * math.log10((2 * vmax) ** 2 / max(mse, 1e-12)))


KNOBS = {
    "field": [
        {"name": "width", "label": "width (plane waves)", "min": 32, "max": 1024,
         "default": 256, "step": 32},
        {"name": "hidden_layers", "label": "hidden sine layers", "min": 1,
         "max": 8, "default": 3, "step": 1},
        # The field's fine component is 15 px per cycle on the full 1024 grid,
        # so about 68 cycles across the frame at downsample 2. omega_0 = 120
        # starts the first layer near 13 cycles and leaves the rest to the
        # layers above; the learning rate below is the measured ceiling for it,
        # omega_0 * lr = 0.03.
        {"name": "omega_0", "label": "omega_0 (x and y)", "min": 5,
         "max": 1000, "default": 120, "step": 1, "log": True},
        # The hash grid caps its time axis in frames per cell. A SIREN has no
        # axis to cap, so the same request is met by stretching t before the
        # first layer: the scale is solved for from the initialised weights so
        # the fastest first-layer wave along t has this period.
        {"name": "frames_per_finest_cycle", "label": "frames per finest cycle",
         "choices": [1, 2, 4, 8, 16], "default": 2},
    ],
    "train": [
        {"name": "lr", "label": "learning rate", "min": 1e-6, "max": 1e-2,
         "default": 2.5e-4, "step": 1e-6, "log": True},
        {"name": "steps", "label": "iterations", "min": 100, "max": 8000,
         "default": 1500, "step": 100},
        {"name": "batch", "label": "batch size (random x, y, t)", "min": 4096,
         "max": 1048576, "default": 262144, "step": 4096},
    ],
}
DEFAULTS = {k["name"]: k.get("default") for g in KNOBS.values() for k in g}
DEFAULTS.update(field=next((k for k in DATASETS if k.startswith("u ")),
                          next(iter(DATASETS), "u 64 + v")),
                downsample=1, coarse_to_fine=0, holdout=1)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>a scalar field in time</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>two PDEs, one siren</h1>
<p class="sub">f(x, y, t) &rarr; a scalar, fitted from a zarr. The runs hold a
<b>coarse slow wave</b> (2 cycles across the frame, correlated with itself for 18
frames) and a <b>fast Kuramoto on four discs</b> (15 px per cycle, on 15% of the
pixels, decorrelating in <b>one</b> frame). They want opposite things, and a SIREN has
one knob for all of it: x, y and t are three inputs to the same plane waves. The
<b>dataset</b> toggle picks which of the three runs to fit, so it is visible which
one gets given up.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is a siren?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button></div>
</div>
<div class="group"><div class="label">&nbsp;</div>
  <div class="seg" id="playseg" style="display:none"><button id="play">play</button>
    <button class="sp" data-s="0.5">x0.5</button><button class="sp" data-s="1"
    aria-pressed="true">x1</button><button class="sp" data-s="2">x2</button>
    <button class="sp" data-s="4">x4</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div id="playnote" class="note"></div>
<div class="setup" id="setup"></div>
<div class="row equal" style="margin-top:14px">
  <div class="panel"><canvas id="c_target0" width="300" height="300"></canvas>
    <div class="cap">target &mdash; first frame</div></div>
  <div class="panel"><canvas id="c_target1" width="300" height="300"></canvas>
    <div class="cap">middle</div></div>
  <div class="panel"><canvas id="c_target2" width="300" height="300"></canvas>
    <div class="cap">last</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_fit0" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit1" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit2" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_err0" width="300" height="300"></canvas>
    <div class="cap">fit &minus; target &mdash; blue/red, fixed at 5% of the
      shown range</div></div>
  <div class="panel"><canvas id="c_err1" width="300" height="300"></canvas>
    <div class="cap">error</div></div>
  <div class="panel"><canvas id="c_err2" width="300" height="300"></canvas>
    <div class="cap">error</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_lev0" width="300" height="300"></canvas>
    <div class="cap">effective band per pixel</div></div>
  <div class="panel"><canvas id="c_lev1" width="300" height="300"></canvas>
    <div class="cap">bands</div></div>
  <div class="panel"><canvas id="c_lev2" width="300" height="300"></canvas>
    <div class="cap">bands</div></div>
</div>
<div class="row grid3" style="margin-top:8px">
  <div class="panel"><canvas id="c_montage" width="300" height="300"></canvas>
    <div class="cap">what each band adds in SPACE, middle frame &mdash; signed,
      fixed at 5% of the shown range</div></div>
  <div class="panel"><canvas id="c_kymo" width="300" height="300"></canvas>
    <div class="cap">what each band adds in TIME &mdash; x across, t down,
      through the middle row</div></div>
</div>
<div id="levlegend" class="note"></div>
<div class="row" style="margin-top:14px">
  <div class="panel"><canvas id="c_curve" width="470" height="300"></canvas>
    <div class="cap">psnr against iteration</div></div>
  <div class="panel"><canvas id="c_frames" width="470" height="300"></canvas>
    <div class="cap">psnr per frame, on a 4x coarser grid &mdash; held-out in amber</div></div>
</div>
<div class="stats" id="stats"></div>
<div class="modal" id="about" onclick="if(event.target===this)closeAbout()">
  <div class="sheet">__ABOUT__</div></div>
</div><script>
const KNOBS=__KNOBS__, DEF=__DEF__, LUTMAX=__LUTMAX__;
const DATASETS=__DATASETS__;
const knob=Object.assign({}, DEF);
const IMG={}; let POLL=null, SEEN=false, LAST=-1, LASTLEV={};
window.addEventListener("unhandledrejection", e=>
  fetch("/api/clienterror?msg="+encodeURIComponent(String(e.reason))));

const C=document.getElementById("controls");
function seg(title, opts, key){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=title;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{ const [txt,val]=Array.isArray(o)?o:[String(o),o];
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", knob[key]===val);
    b.onclick=()=>{ knob[key]=val;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); setup(); };
    s.appendChild(b); });
  g.append(l,s); C.appendChild(g);
}
seg("dataset", DATASETS.map(k=>[k, k]), "field");
seg("downsample", [1,2,4], "downsample");
seg("coarse to fine", [["off",0],["on",1]], "coarse_to_fine");
seg("hold out every", [["none",1],["4th frame",4],["8th",8]], "holdout");

const K=document.getElementById("knobs");
function panel(title, list){
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  K.appendChild(t);
  list.forEach(p=>{
    if(p.choices){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML=`<span>${p.label}</span>`;
      const s=document.createElement("div"); s.className="seg";
      p.choices.forEach(o=>{ const b=document.createElement("button");
        b.textContent=String(o);
        b.setAttribute("aria-pressed", knob[p.name]===o);
        b.onclick=()=>{ knob[p.name]=o;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b));
          setup(); };
        s.appendChild(b); });
      d.append(lab,s); K.appendChild(d); return;
    }
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.pow2?`2^${Math.round(v)} = ${Math.pow(2,Math.round(v)).toLocaleString()}`
                       :(p.log?(+v).toExponential(1)
                              :(p.step<1?(+v).toFixed(3):String(Math.round(v))));
    const raw=v=>p.log?Math.log10(v):v, un=r=>p.log?Math.pow(10,r):+r;
    val.textContent=fmt(knob[p.name]);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=raw(p.min); r.max=raw(p.max); r.step=p.log?0.02:p.step;
    r.value=raw(knob[p.name]);
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${fmt(p.min)}</span><span>${fmt(p.max)}</span>`;
    r.oninput=()=>{ knob[p.name]=un(r.value); val.textContent=fmt(knob[p.name]);
                    setup(); };
    d.append(lab,r,ends); K.appendChild(d);
  });
}
panel("the encoder", KNOBS.field);
panel("training", KNOBS.train);

let PARAMS="";
function setup(){
  document.getElementById("setup").innerHTML=
    `<span class="dim">the</span> <b>${knob.field}</b> <span class="dim">dataset at</span> `
   +`<b>1/${knob.downsample}</b> <span class="dim">resolution, width</span> `
   +`<b>${knob.width}</b> <span class="dim">x</span> <b>${knob.hidden_layers}</b> `
   +`<span class="dim">layers, &omega;<sub>0</sub></span> `
   +`<b>${(+knob.omega_0).toFixed(0)}</b>` + PARAMS;
}
setup();

async function startRun(){
  SEEN=false;
  await fetch("/api/start?"+new URLSearchParams(knob));
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 600); poll();
}
document.getElementById("run").onclick=startRun;
document.getElementById("stop").onclick=()=>fetch("/api/stop");
function openAbout(){ document.getElementById("about").classList.add("on"); }
function closeAbout(){ document.getElementById("about").classList.remove("on"); }

function drawImg(id, src){
  if(!src) return;
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  g.imageSmoothingEnabled=false;
  g.drawImage(im,(cv.width-im.width*s)/2,(cv.height-im.height*s)/2,
              im.width*s, im.height*s);
}
function levColor(t){
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
function drawLevels(id, bk, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!bk||!bk.png) return;
  drawImg(id, bk.png);
  const L=[]; for(let i=0;i<3;i++){ const b=LASTLEV[String(i)]; if(b) L.push(b); }
  if(!L.length) return;
  let bar=""; for(let i=0;i<=LUTMAX;i++)
    bar+=`<span style="display:inline-block;width:14px;height:9px;`
        +`background:${levColor(i/LUTMAX)}"></span>`;
  document.getElementById("levlegend").innerHTML=
    `<div>effective band per pixel &mdash; the band index weighted by how much `
   +`releasing it moved the fit. Fixed 0&ndash;${LUTMAX}.</div>`
   +`<div style="line-height:0;margin:3px 0">${bar}</div>`
   +`<div>frame means ` + L.map(b=>`<b>${b.mean.toFixed(2)}</b>`).join(" / ")
   +`, ranges ` + L.map(b=>`${b.lo.toFixed(1)}&ndash;${b.hi.toFixed(1)}`).join(" / ")
   +`. The fine component sits on 15% of the pixels, so the discs should pull `
   +`this up if the network is spending its high bands where the data is.</div>`;
}
function axes(g,cv,pad,xs,ys,xlab,ylab){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  g.strokeStyle="#333"; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad.t+(cv.height-pad.t-pad.b)*i/4;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(cv.width-pad.r,y); g.stroke(); }
  g.fillStyle="#9a9a9a"; g.font="10px sans-serif"; g.textAlign="center";
  g.fillText(xlab, (pad.l+cv.width-pad.r)/2, cv.height-6);
}
function drawCurve(cur){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const pad={l:44,r:10,t:12,b:22};
  axes(g,cv,pad,null,null,"iteration");
  if(!cur||!cur.length) return;
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const lo=Math.min(...cur.map(q=>q.psnr)), hi=Math.max(...cur.map(q=>q.psnr));
  const X=i=>pad.l+W*i/Math.max(1,cur.length-1);
  const Y=v=>pad.t+H*(1-(v-lo)/Math.max(1e-6,hi-lo));
  g.strokeStyle="#fff"; g.lineWidth=1.6; g.beginPath();
  cur.forEach((q,i)=> i?g.lineTo(X(i),Y(q.psnr)):g.moveTo(X(i),Y(q.psnr)));
  g.stroke();
  g.fillStyle="#9a9a9a"; g.textAlign="right"; g.font="10px sans-serif";
  g.fillText(hi.toFixed(1), pad.l-4, pad.t+8);
  g.fillText(lo.toFixed(1), pad.l-4, pad.t+H);
}
function drawFrames(pf){
  const cv=document.getElementById("c_frames"), g=cv.getContext("2d");
  const pad={l:44,r:10,t:12,b:22};
  axes(g,cv,pad,null,null,"frame");
  if(!pf||!pf.length) return;
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const lo=Math.min(...pf.map(q=>q.psnr)), hi=Math.max(...pf.map(q=>q.psnr));
  const tmax=Math.max(...pf.map(q=>q.t));
  const X=t=>pad.l+W*t/Math.max(1,tmax);
  const Y=v=>pad.t+H*(1-(v-lo)/Math.max(1e-6,hi-lo));
  g.strokeStyle="#4da3ff"; g.lineWidth=1.4; g.beginPath();
  pf.forEach((q,i)=> i?g.lineTo(X(q.t),Y(q.psnr)):g.moveTo(X(q.t),Y(q.psnr)));
  g.stroke();
  g.fillStyle="#e5a23c";
  pf.filter(q=>q.held).forEach(q=>
    g.fillRect(X(q.t)-1.5, Y(q.psnr)-1.5, 3, 3));
  g.fillStyle="#9a9a9a"; g.textAlign="right"; g.font="10px sans-serif";
  g.fillText(hi.toFixed(1), pad.l-4, pad.t+8);
  g.fillText(lo.toFixed(1), pad.l-4, pad.t+H);
}
async function poll(){
  const r=await (await fetch("/api/state")).json();
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";
  const m=r.metrics||{};
  if(m.n_parameters!==undefined && m.n_table!==undefined){
    PARAMS=` <span class="dim">&middot;</span> <span style="color:#fff">`
      +`${m.n_parameters.toLocaleString()} parameters, `
      +`${m.n_table.toLocaleString()} in the first layer, `
      +`<b style="color:${m.compression > 2 ? "#2ea043"
                        : m.compression > 1 ? "#e5a23c" : "#e5484d"}">`
      +`${m.compression.toFixed(1)}x</b> compression against the `
      +`${m.n_values.toLocaleString()} stored values</span>`;
    setup();
  }
  if(r.stamp!==LAST){
    LAST=r.stamp; LASTLEV=r.levels||{};
    for(let i=0;i<3;i++){
      drawImg("c_target"+i, (r.images||{})["target"+i]);
      drawImg("c_fit"+i, (r.images||{})["fit"+i]);
      drawImg("c_err"+i, (r.images||{})["err"+i]);
      drawLevels("c_lev"+i, LASTLEV[String(i)], "c_target"+i);
    }
    drawImg("c_montage", (r.images||{}).montage);
    drawImg("c_kymo", (r.images||{}).kymo);
    drawCurve(r.curve); drawFrames(r.per_frame);
    document.getElementById("stats").innerHTML = m.psnr===undefined ? "press run"
      : `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `
       +`${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; levels live `
       +`<b>${r.levels_live}</b> of ${m.n_levels_total}<br>`
       +`psnr over the three drawn frames, at full resolution `
       +`<b>${m.psnr.toFixed(2)}</b> dB`
       + (m.psnr_held ? ` &nbsp;&middot;&nbsp; on the <b>${m.n_held}</b> `
                       +`held-out frames <b>${m.psnr_held.toFixed(2)}</b> dB` : "")
       +`<br><span style="color:#7a7a7a">${r.note}</span>`;
  }
  showPlay(!r.running && r.step > 0);
  if(r.running) SEEN=true;
  if(SEEN && !r.running && POLL){ clearInterval(POLL); POLL=null; }
}
poll();
// The play button: the run frame by frame, target beside fit, in the two left
// panels.  Hidden until a fit has finished, because there is nothing to infer
// from an untrained model and a half-trained one is already on the panels.
let PLAYING=false, PLAYAT=0, PLAYSPEED=1;
// Speed moves BOTH the timer and the stride. The server renders a frame in
// about 40 ms, so asking for them faster than that just queues; skipping frames
// is what actually makes the run play quicker.
document.querySelectorAll("#playseg .sp").forEach(b=>{
  b.onclick=()=>{ PLAYSPEED=parseFloat(b.dataset.s);
    document.querySelectorAll("#playseg .sp").forEach(o=>
      o.setAttribute("aria-pressed", o===b)); };
});
function showPlay(on){
  // The GROUP, not the button: a hidden button inside the seg is still its
  // last-child, and .seg button:last-child is what draws the right-hand border,
  // so hiding it that way cut the edge off the button before it.
  document.getElementById("playseg").style.display = on ? "" : "none";
}
async function playStep(){
  if(!PLAYING) return;
  let r;
  try { r = await (await fetch("/api/frame?i="+PLAYAT)).json(); }
  catch(e){ PLAYING=false; return; }
  if(r.error){ PLAYING=false; document.getElementById("play").textContent="play";
               return; }
  drawImg("c_target0", r.target);
  drawImg("c_fit0", r.fit);
  drawImg("c_err0", r.resid);
  document.getElementById("playnote").textContent =
    `frame ${r.i + 1} / ${r.n}` + (PLAYSPEED === 1 ? "" :
      `   x${PLAYSPEED}, ${1 * Math.max(1, Math.round(PLAYSPEED))} `
      + `frames per step`);
  PLAYAT = (r.i + 1) % r.n;
  setTimeout(playStep, 40);
}
document.getElementById("play").onclick=()=>{
  PLAYING = !PLAYING;
  document.getElementById("play").textContent = PLAYING ? "pause" : "play";
  if(PLAYING) playStep();
};
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
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            page = (PAGE.replace("__CSS__", CSS)
                        .replace("__ABOUT__", ABOUT_HTML)
                        .replace("__KNOBS__", json.dumps(KNOBS))
                        .replace("__DEF__", json.dumps(DEFAULTS))
                        # Whatever is on disk right now, sum first. Filtering
                        # against a fixed list of names is what emptied this menu
                        # when the labels gained their resolution.
                        .replace("__DATASETS__",
                                 json.dumps(sorted(datasets(ZARR_GLOB),
                                                   key=lambda k:
                                                   (not k.startswith("u "), k))))
                        .replace("__LUTMAX__", str(LEVEL_LUT_MAX))
                        .replace("__ERRMAX__", f"{ERROR_LUT_MAX:g}")
)
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            with LOCK:
                JOB["running"] = True
                JOB["stamp"] += 1
            # Rescan: the store may have appeared, moved or been regenerated
            # since the page was opened.
            found = datasets(ZARR_GLOB)
            DATASETS.clear()
            DATASETS.update(found)
            p = dict(DEFAULTS)
            for k, v in q.items():
                p[k] = float(v) if _isnum(v) else v
            threading.Thread(target=train_job, args=(p, self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            STOP.set()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clienterror":
            print(f"[client] {q.get('msg', '')}", flush=True)
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/frame":
            # One frame, target and fit, rendered on demand: 201 frames of two
            # panels is a lot of png to send for something watched once.
            if not PLAY:
                return self._send(json.dumps({"error": "no finished fit"}),
                                  "application/json")
            i = max(0, min(PLAY["n_frames"] - 1, int(float(q.get("i", 0)))))
            tgt, fit, resid = _frame_pngs(i)
            return self._send(json.dumps(
                {"i": i, "n": PLAY["n_frames"], "target": tgt, "fit": fit,
                 "resid": resid, "ready": PLAY.get("ready", 0)}),
                "application/json")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8125)
    ap.add_argument("--zarr-glob", default=None, action="append",
                    help="where to look for stores; repeatable, and replaces "
                         "the two defaults (the toy generator's log and this "
                         "repo's data/)")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.zarr_glob = a.zarr_glob or ZARR_GLOB
    globals()["ZARR_GLOB"] = a.zarr_glob
    DATASETS.clear()
    DATASETS.update(datasets(a.zarr_glob))
    # NOT a hard exit when nothing is found. The generator rewrites this tree
    # while it runs -- the directory this page was written against was renamed
    # twice in one afternoon -- so the page comes up either way and rescans on
    # every run, and pressing run once the store lands is enough.
    if not DATASETS:
        pats = ([a.zarr_glob] if isinstance(a.zarr_glob, str) else a.zarr_glob)
        near = sorted({os.path.basename(os.path.dirname(d))
                       for pat in pats
                       for d in glob.glob(os.path.dirname(
                           os.path.dirname(pat)) + "/*")})
        print(f"no store under {a.zarr_glob} yet -- the page will rescan on "
              f"every run. Beside it: {', '.join(near[:8]) or 'nothing'}",
              flush=True)
    # "sum" even when nothing is on disk: the page rescans on every run, so the
    # selector has to hold a name until a store appears.
    DEFAULTS["field"] = next((k for k in DATASETS if k.startswith("u ")),
                             sorted(DATASETS)[0] if DATASETS else "u 64 + v")
    Handler.device = torch.device(a.device)
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except OSError as e:
        if e.errno == 98:
            sys.exit(f"port {a.port} is already in use")
        raise
    for k in ("sum", "coarse", "fine"):
        if k in DATASETS:
            print(f"  {k:7s} {os.path.basename(os.path.dirname(DATASETS[k]))}")
    print(f"http://localhost:{a.port}   (device {a.device})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
