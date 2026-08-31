#!/usr/bin/env python
"""Fit an NGP to a field that moves: one (x, y, t) encoder over a whole sequence.

    python scripts/gui_time.py            # http://localhost:8024

The registration page recovers one warp from one pair. This one recovers a warp
that *evolves*: the painting drifts or rotates over N frames, and a single
encoder over (x, y, t) has to hold the whole sequence at once. That is the shape
of the zapbench problem -- a slowly changing deformation over thousands of
frames -- reduced to something that fits on a screen.

Three panels per row, at the first, middle and last frame, because the failure
mode is not visible in any single one: a fit can be excellent where it was
trained hardest and drift at the ends, or hold the ends and sag in between. The
curve underneath scores every frame, so which of those is happening is a fact
rather than an impression.

Motion is stated PER FRAME, so a longer sequence at the same speed travels
further -- which is what a longer recording does, and is why sweeping the frame
count with the speed held is the interesting comparison.
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren.deform import (MovingBand, build_pyramid, pixel_grid, pyramid_level,
                        sample_bilinear)
from siren.siren import SirenField
from siren.utils import psnr
from siren.webui import ABOUT_HTML, CSS, cmap_png, gray_png
from scripts.run_registration import _feather, foreground_mask, load_image, sample_points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPE_LUT_MAX = 10.0                  # fixed, so the error panels mean one thing throughout
RESID_LUT_MAX = 0.10                # the photometric residual, in intensity units

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "note": "", "stamp": 0, "frames": [],
       "per_frame": [], "levels_live": None, "grids": {}, "levels": {}}
LOCK = threading.Lock()
STOP = threading.Event()
SCENE: dict = {}


def get_scene(cfg, device):
    if "source" in SCENE:
        return SCENE
    src = load_image(cfg["image"], device)
    shape = tuple(src.shape[:2])
    fgc = cfg["image"]["foreground"]
    fg = foreground_mask(src, fgc)
    if fgc.get("zero_background"):
        src = src * _feather(fg, fgc.get("feather_px", 9))
    SCENE.update(source=src, shape=shape, fg=fg,
                 fg_idx=torch.nonzero(fg.reshape(-1), as_tuple=False).squeeze(1))
    return SCENE


def train_job(cfg, p, device):
    try:
        sc = get_scene(cfg, device)
        src, shape = sc["source"], sc["shape"]
        h, w = shape
        px = torch.tensor([w, h], device=device, dtype=torch.float32)
        n_frames = int(p["frames"])
        gt = MovingBand(p["motion"], float(p["total"]), n_frames, shape,
                        width_px=float(p["band_width"]),
                        offset_px=float(p["band_offset"]), device=device)

        torch.manual_seed(0)
        # (x, y, t) in, displacement out.  There is no time AXIS to cap here --
        # the whole point of ngp-demo's stage 2 was that the grid's t axis has to
        # be capped at the frame spacing, and a SIREN has no axis to cap: t is
        # just a third input to the same plane waves, so the frequency it is
        # resolved at is omega_0 |w_i,t| like everything else.
        model = SirenField(
            n_input_dims=3, n_output_dims=2, output_activation="none",
            width=int(p["width"]), hidden_layers=int(p["hidden_layers"]),
            omega_0=float(p["omega_0"]), outermost_linear=True,
        ).to(device)
        n_enc, n_mlp = model.n_parameters()
        n_bands = 16
        unit = "px" if p["motion"] == "translate" else "deg"
        note = (f"{n_frames} frames, a {gt.offset:.0f} px slip band {p['motion']}s "
                f"{gt.total():.0f} {unit} in total = {gt.speed:.4g} {unit}/frame, "
                f"band width {gt.width:.0f} px; width {int(p['width'])} x "
                f"{int(p['hidden_layers'])} layers, omega_0 "
                f"{float(p['omega_0']):g}")
        picks = [0, n_frames // 2, n_frames - 1]
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], per_frame=[], note=note, frames=picks,
                       metrics={"n_parameters": n_enc + n_mlp,
                                "n_table": n_enc, "n_decoder": n_mlp,
                                "n_levels_total": n_bands,
                                "total_motion": gt.total(), "unit": unit},
                       images={f"target{i}": gray_png(
                           warp_t(src, gt, t / max(1, n_frames - 1), shape))
                           for i, t in enumerate(picks)},
                       stamp=JOB["stamp"] + 1)
        print(f"[run] {note}", flush=True)
        print(f"[params] {n_enc + n_mlp:,} total = {n_enc:,} in the first layer "
              f"({int(p['width'])} plane waves over x, y and t) "
              f"+ {n_mlp:,} in the decoder", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps = int(p["steps"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                           eta_min=float(p["lr"]) * 0.03)
        pyr = {"sigma_px": [16, 8, 4, 2, 0], "switch_at": [0.0, .15, .30, .45, .60]}
        if not int(p.get("pyramid", 1)):
            pyr = {"sigma_px": [0], "switch_at": [0.0]}
        src_p = build_pyramid(src, pyr["sigma_px"])
        every = max(1, steps // 30)
        t0 = time.perf_counter()
        secs = 0.0

        for step in range(steps + 1):
            if STOP.is_set():
                break
            s0 = time.perf_counter()
            a = 4 + (n_bands - 4) * min(1.0, step / max(1, steps * 0.5))
            model.set_band_window(a, n_bands)
            JOB["levels_live"] = round(float(a), 2)
            lvl = pyramid_level(step, steps, pyr["switch_at"])
            img = src_p[lvl]

            xy = sample_points(sc["fg_idx"], int(p["batch"]), 0.9, shape, device)
            fi = torch.randint(n_frames, (xy.shape[0],), device=device).float()
            tt = (fi / max(1, n_frames - 1)).unsqueeze(1)
            xyt = torch.cat([xy, tt], dim=1)
            u = model(xyt)
            # the target at that instant, generated analytically -- no need to
            # hold N warped copies of the painting in memory
            ugt = gt(xy, tt)                    # analytic, one t per sampled point
            pred = sample_bilinear(img, xy + u / px)
            tgt = sample_bilinear(img, xy + ugt / px)
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            secs += time.perf_counter() - s0

            if step % every == 0 or step == steps:
                m, images, per_frame, grids, levels = evaluate(
                    model, gt, sc, n_frames, device, picks)
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = secs
                    JOB["curve"].append({"step": step, "epe": m["epe_mean"],
                                         "epe_first": per_frame[0][1],
                                         "epe_last": per_frame[-1][1]})
                    JOB["per_frame"] = per_frame
                    JOB["metrics"].update(m)
                    JOB["images"].update(images)
                    JOB["grids"] = grids
                    JOB["levels"] = levels
                    JOB["stamp"] += 1
    except Exception as e:
        print(f"[run] failed: {type(e).__name__}: {e}", flush=True)
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1
            m = JOB["metrics"]
        print(f"[{'done ' if JOB['step'] >= JOB['steps'] > 0 else 'stopped'}] "
              f"{JOB['seconds']:.1f}s  mean EPE {m.get('epe_mean', float('nan')):.3f} px "
              f"(first {m.get('epe_first', float('nan')):.3f}, "
              f"last {m.get('epe_last', float('nan')):.3f})", flush=True)


def warp_t(src, gt, t, shape, chunk=1 << 20):
    from siren.deform import warp_image
    return warp_image(src, gt.at(t), shape, chunk)


@torch.no_grad()
def dense_u(model, shape, t, device, chunk=1 << 18):
    xy = pixel_grid(*shape, device)
    tt = torch.full((xy.shape[0], 1), float(t), device=device)
    xyt = torch.cat([xy, tt], dim=1)
    return torch.cat([model(xyt[i:i + chunk]) for i in range(0, len(xyt), chunk)])


@torch.no_grad()
def band_map_at(model, shape, device, t, sub=4):
    """Per pixel, the effective frequency band of u at time t.

    Render u with bands 0..k enabled for every k, difference consecutive
    renders, and weight the band index by how much each release moved the field.
    Watching it across three frames answers whether the network's use of
    frequency follows the slip band as the slip band moves.
    """
    n_bands = 16
    f = model.frequencies().cpu()
    bnd = model.band_of(n_bands).cpu()
    h, w = shape
    hs, ws = max(8, h // sub), max(8, w // sub)
    xy = pixel_grid(hs, ws, device)
    tt = torch.full((xy.shape[0], 1), float(t), device=device)
    xyt = torch.cat([xy, tt], dim=1)
    prev, deltas = None, []
    for k in range(n_bands + 1):
        model.set_band_window(float(k), n_bands)
        out = model(xyt).reshape(hs, ws, 2)
        if prev is not None:
            deltas.append((out - prev).norm(dim=-1))
        prev = out
    model.set_band_window(float(n_bands), n_bands)
    D = torch.stack(deltas)
    # THE EFFECTIVE BAND PER PIXEL, not the finest band per block.  A grid level
    # only touches its own cells, so a block can name the one that moved it; a
    # SIREN band is a plane wave over the whole frame and every band contributes
    # everywhere -- measured on this benchmark, the busiest tenth of the blocks
    # holds 18-33% of any band's total (10% is uniform), and the block version
    # named the finest band in 239 of 255 blocks.  The amplitude-weighted mean
    # band does vary, and it answers the question this row is for: does the
    # network lean on higher frequencies where the slip band is?
    b = torch.arange(D.shape[0], device=D.device, dtype=D.dtype)
    eff = (D * b[:, None, None]).sum(0) / D.sum(0).clamp(min=1e-9)
    return {"png": cmap_png(eff.cpu().numpy(), n_bands - 1, "levels"),
            "mean": float(eff.mean()), "lo": float(eff.min()),
            "hi": float(eff.max()), "n_levels": n_bands}


def grid_lines(u, spacing, shape):
    """A regular grid carried through x -> x + u(x), as polylines in image px."""
    h, w = shape
    ys, xs = np.arange(0, h, spacing), np.arange(0, w, spacing)
    out = []
    for y in ys:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for x in xs])
    for x in xs:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for y in ys])
    return out


def evaluate(model, gt, sc, n_frames, device, picks, n_probe=25):
    src, shape, fg = sc["source"], sc["shape"], sc["fg"]
    h, w = shape
    xy = pixel_grid(h, w, device)
    sel = fg.reshape(-1)
    images, per_frame, grids, levels = {}, [], {}, {}
    # every frame is scored, not only the three shown: the whole question is
    # whether the fit holds across time or only where it was pushed hardest
    probe = np.unique(np.linspace(0, n_frames - 1, n_probe).round().astype(int))
    for f in probe:
        t = f / max(1, n_frames - 1)
        u = dense_u(model, shape, t, device)
        ugt = gt(xy, t)
        epe = (u - ugt).norm(dim=1)
        per_frame.append((int(f), float(epe[sel].mean())))
        if int(f) in picks:
            i = picks.index(int(f))
            warped = warp_model(src, model, shape, t, device)
            images[f"fit{i}"] = gray_png(warped)
            # what the LOSS sees, as opposed to what the field got wrong: a fit
            # can match the picture and still have the field wrong wherever the
            # picture has no gradient
            tgt = warp_t(src, gt, t, shape)
            images[f"resid{i}"] = cmap_png((warped - tgt).abs().cpu().numpy(),
                                           RESID_LUT_MAX)
            images[f"epe{i}"] = cmap_png(epe.reshape(h, w).cpu().numpy(), EPE_LUT_MAX)
            # the field itself, both versions -- everything else on the page is a
            # consequence of the deformation rather than the deformation
            un = u.reshape(h, w, 2).cpu().numpy()
            gn = ugt.reshape(h, w, 2).cpu().numpy()
            # exposed for THIS run's motion, so the field is legible whatever
            # the total is set to
            levels[str(i)] = band_map_at(model, shape, device, t)
            grids[str(i)] = {"gt": grid_lines(gn, 64, shape),
                             "fit": grid_lines(un, 64, shape), "w": w, "h": h}
    e = np.array([v for _, v in per_frame])
    m = {"epe_mean": float(e.mean()), "epe_max": float(e.max()),
         "epe_first": per_frame[0][1], "epe_last": per_frame[-1][1],
         "epe_mid": per_frame[len(per_frame) // 2][1]}
    return m, images, per_frame, grids, levels


@torch.no_grad()
def warp_model(src, model, shape, t, device, chunk=1 << 20):
    h, w = shape
    xy = pixel_grid(h, w, device)
    px = torch.tensor([w, h], device=device, dtype=torch.float32)
    out = []
    for i in range(0, xy.shape[0], chunk):
        q = xy[i:i + chunk]
        tt = torch.full((q.shape[0], 1), float(t), device=device)
        out.append(sample_bilinear(src, q + model(torch.cat([q, tt], 1)) / px))
    return torch.cat(out).reshape(h, w)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>siren over a moving field</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>one encoder over (x, y, t) &mdash; a field that moves</h1>
<p class="sub">A slip band travels or turns across the painting over N frames, and a
single hash grid holds the whole sequence. The deformation PATTERN moves: the band sits
somewhere different in every frame, so the encoder has to put its detail somewhere
different too. Three columns: the first frame, the middle one, the last.
A fit can be excellent where it was pushed hardest and drift at the ends, or hold
the ends and sag between them, and neither shows in one frame &mdash; the curve
scores every frame so which one is happening is a fact.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is a siren?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="setup" id="setup"></div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_target0" width="300" height="380"></canvas>
    <div class="cap" id="cap0">first frame &mdash; target</div></div>
  <div class="panel"><canvas id="c_target1" width="300" height="380"></canvas>
    <div class="cap" id="cap1">middle frame &mdash; target</div></div>
  <div class="panel"><canvas id="c_target2" width="300" height="380"></canvas>
    <div class="cap" id="cap2">last frame &mdash; target</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_fit0" width="300" height="380"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit1" width="300" height="380"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit2" width="300" height="380"></canvas>
    <div class="cap">fit</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_epe0" width="300" height="380"></canvas>
    <div class="cap">endpoint error &mdash; fixed 0&ndash;10 px</div></div>
  <div class="panel"><canvas id="c_epe1" width="300" height="380"></canvas>
    <div class="cap">endpoint error</div></div>
  <div class="panel"><canvas id="c_epe2" width="300" height="380"></canvas>
    <div class="cap">endpoint error</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_resid0" width="300" height="380"></canvas>
    <div class="cap">image residual &mdash; fixed 0&ndash;0.1</div></div>
  <div class="panel"><canvas id="c_resid1" width="300" height="380"></canvas>
    <div class="cap">image residual</div></div>
  <div class="panel"><canvas id="c_resid2" width="300" height="380"></canvas>
    <div class="cap">image residual</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_lev0" width="300" height="380"></canvas>
    <div class="cap">the pyramid in use &mdash; finest level contributing per block</div></div>
  <div class="panel"><canvas id="c_lev1" width="300" height="380"></canvas>
    <div class="cap">effective band per pixel</div></div>
  <div class="panel"><canvas id="c_lev2" width="300" height="380"></canvas>
    <div class="cap">effective band per pixel</div></div>
</div>
<div id="levlegend" class="note"></div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_grid0" width="300" height="380"></canvas>
    <div class="cap">grid &mdash; <i>ground truth</i> vs <b>fit</b></div></div>
  <div class="panel"><canvas id="c_grid1" width="300" height="380"></canvas>
    <div class="cap">grid</div></div>
  <div class="panel"><canvas id="c_grid2" width="300" height="380"></canvas>
    <div class="cap">grid</div></div>
</div>
<div class="row" style="margin-top:18px">
  <div class="panel"><canvas id="c_frames" width="620" height="300"></canvas>
    <div class="cap">endpoint error at every frame &mdash; the three shown are marked</div></div>
  <div class="panel"><canvas id="c_curve" width="420" height="300"></canvas>
    <div class="cap">endpoint error against iteration</div></div>
</div>
<div class="stats" id="stats"></div>
<div class="modal" id="about" onclick="if(event.target===this)closeAbout()">
  <div class="sheet">__ABOUT__</div></div>
</div><script>
function openAbout(){document.getElementById("about").classList.add("open");}
function closeAbout(){document.getElementById("about").classList.remove("open");}
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeAbout();});
function _report(w){ try{ fetch("/api/clienterror?msg="+encodeURIComponent(String(w).slice(0,800))); }catch(e){} }
window.onerror=(m,s,l,c,err)=>_report((err&&err.stack)||(m+" @"+l+":"+c));
window.addEventListener("unhandledrejection",e=>_report("unhandled: "+((e.reason&&e.reason.stack)||e.reason)));

const KNOBS=__KNOBS__, DEF=__DEF__;
const knob=Object.assign({}, DEF);
let LAST=-1, POLL=null, IMG={}, SEEN_RUNNING=false;

const C=document.getElementById("controls");
function seg(name, opts, key){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(([txt,val])=>{
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", knob[key]===val);
    b.onclick=()=>{ knob[key]=val;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b));
      PARAMS=""; setup(); };
    s.appendChild(b); });
  g.append(l,s); C.appendChild(g);
}
seg("the band", [["translates","translate"],["rotates","rotate"]], "motion");
seg("frames", [["100",100],["200",200],["500",500],["800",800]], "frames");
seg("image pyramid", [["on",1],["off",0]], "pyramid");

const K=document.getElementById("knobs");
function panel(title, list){
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  K.appendChild(t);
  list.forEach(p=>{
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.log?(+v).toExponential(1):(p.step<1?(+v).toFixed(3):String(Math.round(v)));
    const raw=v=>p.log?Math.log10(v):v, un=r=>p.log?Math.pow(10,r):+r;
    val.textContent=fmt(knob[p.name]);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=raw(p.min); r.max=raw(p.max); r.step=p.log?0.02:p.step; r.value=raw(knob[p.name]);
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${fmt(p.min)}</span><span>${fmt(p.max)}</span>`;
    r.oninput=()=>{ knob[p.name]=un(r.value); val.textContent=fmt(knob[p.name]);
                    PARAMS=""; setup(); };
    d.append(lab,r,ends); K.appendChild(d);
  });
}
panel("the motion and the encoder", KNOBS.model);
panel("training", KNOBS.train);

// The parameter count belongs on the setup line and not in the metrics: it is a
// property of the SETTINGS, fixed before the first step. Cleared on any knob
// edit, because from that moment it describes the run in flight and not the one
// the sentence above now describes.
let PARAMS="";
function setup(){
  const unit = knob.motion==="translate" ? "px" : "deg";
  const per = knob.total/Math.max(1, knob.frames-1);
  document.getElementById("setup").innerHTML=
    `<span class="dim">a</span> <b>${(+knob.band_offset).toFixed(0)} px</b> `
    +`<span class="dim">slip band that</span> <b>${knob.motion}s</b> `
    +`<b>${(+knob.total).toFixed(0)} ${unit}</b> <span class="dim">over</span> `
    +`<b>${knob.frames}</b> <span class="dim">frames =</span> `
    +`<b>${per.toFixed(4)} ${unit}/frame</b>`+PARAMS;
}
setup();

async function startRun(){
  SEEN_RUNNING=false;
  await fetch("/api/start?"+new URLSearchParams(knob));
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 500); poll();
}
document.getElementById("run").onclick=startRun;
document.getElementById("stop").onclick=()=>fetch("/api/stop");

function drawImg(id, src){
  const cv=document.getElementById(id); if(!cv) return;
  const g=cv.getContext("2d");
  if(!src){ g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height); return; }
  if(IMG[id]&&IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  g.drawImage(im,(cv.width-im.width*s)/2,(cv.height-im.height*s)/2,im.width*s,im.height*s);
}

function levColor(t){
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
let LASTLEV={};
const LUTMAX=15;
function drawLevels(id, bk, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!bk||!bk.png) return;
  drawImg(id, bk.png);
  const L=[];
  for(let i=0;i<3;i++){ const b=(LASTLEV||{})[String(i)]; if(b) L.push(b); }
  if(!L.length) return;
  let bar=""; for(let i=0;i<=LUTMAX;i++)
    bar+=`<span style="display:inline-block;width:14px;height:9px;`
        +`background:${levColor(i/LUTMAX)}"></span>`;
  document.getElementById("levlegend").innerHTML=
    `<div>effective band per pixel &mdash; the band index weighted by how much `
    +`releasing it moved the field. Fixed scale 0&ndash;${LUTMAX}.</div>`
    +`<div style="line-height:0;margin:3px 0">${bar}</div>`
    +`<div>frame means ` + L.map(b=>`<b>${b.mean.toFixed(2)}</b>`).join(" / ")
    +`, ranges ` + L.map(b=>`${b.lo.toFixed(1)}&ndash;${b.hi.toFixed(1)}`).join(" / ")
    +`. A SIREN band is a plane wave over the whole frame, so "the finest band in `
    +`this block" has no answer &mdash; that map named the top band nearly `
    +`everywhere. This one is what does vary.</div>`;
}
function drawGrid(id, gr){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!gr||!gr.gt) return;
  const s=Math.min(cv.width/gr.w, cv.height/gr.h);
  const ox=(cv.width-gr.w*s)/2, oy=(cv.height-gr.h*s)/2;
  const paint=(L,col,lw,dash)=>{ g.strokeStyle=col; g.lineWidth=lw;
    g.setLineDash(dash||[]);
    L.forEach(p=>{ g.beginPath();
      p.forEach((q,i)=>{ const X=ox+q[0]*s, Y=oy+q[1]*s; i?g.lineTo(X,Y):g.moveTo(X,Y); });
      g.stroke(); }); g.setLineDash([]); };
  paint(gr.gt, "#e5484d", 1.2);
  paint(gr.fit, "#4da3ff", 1.0, [4,3]);
}
function drawFrames(pf, picks){
  const cv=document.getElementById("c_frames"), g=cv.getContext("2d");
  const W=cv.width,H=cv.height,pad={l:56,r:12,t:14,b:34};
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  if(!pf||pf.length<2) return;
  const xs=pf.map(p=>p[0]), ys=pf.map(p=>p[1]);
  const x1=Math.max(...xs)||1, lo=0, hi=Math.max(...ys)*1.1||1;
  const X=v=>pad.l+v/x1*(W-pad.l-pad.r), Y=v=>pad.t+(1-v/hi)*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.fillStyle="#666"; g.font="10px monospace";
  for(let i=0;i<=4;i++){ const v=hi*i/4,y=Y(v);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.fillText(v.toFixed(2), 6, y+3); }
  (picks||[]).forEach(f=>{ const x=X(f);
    g.strokeStyle="#4a4a4a"; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(x,pad.t); g.lineTo(x,H-pad.b); g.stroke(); g.setLineDash([]); });
  g.strokeStyle="#4da3ff"; g.lineWidth=2; g.beginPath();
  pf.forEach((p,i)=>{ const x=X(p[0]),y=Y(p[1]); i?g.lineTo(x,y):g.moveTo(x,y); });
  g.stroke();
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  g.fillText("frame", W/2-16, H-10);
  g.save(); g.translate(14,H/2+40); g.rotate(-Math.PI/2);
  g.fillText("endpoint error (px)",0,0); g.restore();
}

function drawCurve(c){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const W=cv.width,H=cv.height,pad={l:52,r:12,t:14,b:34};
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  if(!c||c.length<2) return;
  const vals=c.flatMap(q=>[q.epe,q.epe_first,q.epe_last]).filter(v=>v>0);
  const lo=Math.max(1e-4,Math.min(...vals)), hi=Math.max(...vals);
  const x1=Math.max(...c.map(q=>q.step))||1;
  const X=v=>pad.l+v/x1*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/
                    ((Math.log10(hi)-Math.log10(lo))||1))*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.font="10px monospace";
  for(let d=Math.floor(Math.log10(lo)); d<=Math.ceil(Math.log10(hi)); d++){
    const y=Y(Math.pow(10,d)); if(y<pad.t||y>H-pad.b) continue;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.fillText("1e"+d, 6, y+3); }
  const line=(k,col)=>{ g.strokeStyle=col; g.lineWidth=1.8; g.beginPath();
    c.forEach((q,i)=>{ const x=X(q.step),y=Y(q[k]); i?g.lineTo(x,y):g.moveTo(x,y); });
    g.stroke(); };
  line("epe","#e8e8e8"); line("epe_first","#2ea043"); line("epe_last","#e5a23c");
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  [["all frames","#e8e8e8"],["first","#2ea043"],["last","#e5a23c"]].forEach(([t,col],i)=>{
    g.fillStyle=col; g.fillRect(W-130, pad.t+i*14-8, 9, 2);
    g.fillStyle="#9a9a9a"; g.fillText(t, W-116, pad.t+i*14-3); });
}

async function poll(){
  const r=await (await fetch("/api/state")).json();
  document.getElementById("prog").style.width=(r.steps?(r.step/r.steps*100):0)+"%";
  if(r.running) SEEN_RUNNING=true;
  if(r.stamp!==LAST){
    LAST=r.stamp;
    for(let i=0;i<3;i++){
      drawImg("c_target"+i, r.images["target"+i]);
      drawImg("c_fit"+i, r.images["fit"+i]);
      drawImg("c_epe"+i, r.images["epe"+i]);
      drawImg("c_resid"+i, r.images["resid"+i]);
      LASTLEV=r.levels||{};
      drawLevels("c_lev"+i, LASTLEV[String(i)], "c_target"+i);
      drawGrid("c_grid"+i, (r.grids||{})[String(i)]);
      const cap=document.getElementById("cap"+i);
      if(cap && r.frames && r.frames.length===3)
        cap.textContent=["first","middle","last"][i]+" frame "+r.frames[i]+" — target";
    }
    drawFrames(r.per_frame, r.frames); drawCurve(r.curve);
    const m=r.metrics||{};
  if(m.n_parameters!==undefined && m.n_table!==undefined){
    const np=` <span class="dim">&middot;</span> <span style="color:#fff">`
      +`${m.n_parameters.toLocaleString()} parameters, `
      +`${m.n_table.toLocaleString()} in the hash table `
      +`+ ${m.n_decoder.toLocaleString()} in the decoder</span>`;
    if(np!==PARAMS){ PARAMS=np; setup(); }
  }
    document.getElementById("stats").innerHTML = m.epe_mean===undefined
      ? (r.running?"running&hellip;":"")
      : `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `
       +`${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; levels live `
       +`<b>${r.levels_live}</b> of ${m.n_levels_total}<br>`
       +`endpoint error &nbsp; mean over all frames <b>${m.epe_mean.toFixed(3)}</b> px`
       +` &nbsp; worst frame <b>${m.epe_max.toFixed(3)}</b>`
       +` &nbsp;&middot;&nbsp; first <b>${m.epe_first.toFixed(3)}</b>`
       +` &nbsp; middle <b>${m.epe_mid.toFixed(3)}</b>`
       +` &nbsp; last <b>${m.epe_last.toFixed(3)}</b><br>`
       +`total motion <b>${m.total_motion.toFixed(1)} ${m.unit}</b>`
       +`<br><span style="color:#7a7a7a">${r.note}</span>`;
  }
  if(SEEN_RUNNING && !r.running && POLL){ clearInterval(POLL); POLL=null; }
}
poll(); startRun();
</script></body></html>
"""

KNOB_SPEC = {
    "model": [
        {"name": "total", "label": "how far the band moves over the sequence (px or deg)",
         "min": 10, "max": 720, "default": 300, "step": 5, "log": True},
        {"name": "band_offset", "label": "slip across the band (px)", "min": 2,
         "max": 60, "default": 18, "step": 1},
        {"name": "band_width", "label": "band width (px)", "min": 2, "max": 120,
         "default": 12, "step": 1},
        {"name": "width", "label": "width (plane waves)", "min": 32, "max": 1024,
         "default": 256, "step": 32},
        {"name": "hidden_layers", "label": "hidden sine layers", "min": 1,
         "max": 8, "default": 3, "step": 1},
        # One omega_0 for all three inputs, x, y and t alike. The grid twin has
        # to cap its time axis at the frame spacing or it memorises frames; a
        # SIREN has no axis to cap, so this is the only place the time scale
        # enters.
        {"name": "omega_0", "label": "omega_0", "min": 5, "max": 500,
         "default": 30, "step": 1, "log": True},
    ],
    "train": [
        {"name": "lr", "label": "learning rate", "min": 1e-6, "max": 1e-1,
         "default": 5e-4, "step": 1e-6, "log": True},
        {"name": "steps", "label": "iterations", "min": 200, "max": 6000,
         "default": 1500, "step": 100},
        {"name": "batch", "label": "batch size (sample points)", "min": 4096,
         "max": 262144, "default": 65536, "step": 4096},
    ],
}
KNOB_DEFAULTS = {k["name"]: k["default"] for grp in KNOB_SPEC.values() for k in grp}
KNOB_DEFAULTS.update(motion="translate", frames=200, pyramid=1)


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
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            page = (PAGE.replace("__CSS__", CSS).replace("__ABOUT__", ABOUT_HTML)
                        .replace("__KNOBS__", json.dumps(KNOB_SPEC))
                        .replace("__DEF__", json.dumps(KNOB_DEFAULTS)))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/clienterror":
            print(f"[client] {q.get('msg','')}", flush=True)
            return self._send("{}", "application/json")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            with LOCK:
                JOB["running"] = True
                JOB["stamp"] += 1
            p = dict(KNOB_DEFAULTS)
            p.update({k: (float(v) if _isnum(v) else v) for k, v in q.items()})
            threading.Thread(target=train_job, args=(self.cfg, p, self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
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


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config/registration_benchmark.yaml"))
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    with open(a.config) as f:
        Handler.cfg = yaml.safe_load(f)
    Handler.device = torch.device(a.device)
    print(f"http://localhost:{a.port}   (device {a.device})")
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except OSError as e:
        if e.errno == 98:
            sys.exit(f"port {a.port} is already in use -- pass --port with a free one")
        raise
    srv.serve_forever()


if __name__ == "__main__":
    main()
