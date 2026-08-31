# siren_demo — a SIREN taken apart while it trains

The twin of [ngp-demo](https://github.com/allierc/ngp-demo), for the other kind of
implicit representation: no grid, no hash table, no levels — a plain MLP whose
activation is `sin`, where the detail lives in the frequencies.

Same painting, same benchmarks, same metrics, same three browser pages, so the two
repos can be run side by side and their numbers compared directly. Everything below
was measured by the scripts in this repo, on an RTX A6000, torch 2.9.0+cu128.

```
siren/siren.py    SineLayer / Siren, vendored from connectome-gnn (which adapted
                  them from vsitzmann/siren), plus SirenField: the frequency
                  ladder, the band window, jacobian() / laplacian()
siren/deform.py   ground-truth warps, SirenDeform, the control-grid baseline
siren/webui.py    shared CSS, PNG helpers, and the two explainers
scripts/          gui_image.py, gui_field.py, gui_time.py, run_registration.py
tests/            test_siren.py (8 checks), check_pages.py
```

## The idea the pages are built on

A SIREN is usually described as having no scales. It has them, in the open:

```
first-layer unit i:   sin( omega_0 * (w_i . x + b_i) )        a plane wave
frequency_i        =  omega_0 * |w_i| / 2pi                   cycles across the frame
px per cycle       =  W_px / frequency_i

gate:  h = g * sin(...)      g in R^width      <- the twin of the grid's level_gain
```

Sort the units by `frequency_i`, cut into 16 equal-count bands, and one gain vector
gives everything ngp-demo builds on `level_gain`: a band montage, an
effective-scale map per pixel, a ladder table, and a coarse-to-fine window.

Two things are true here that are not true of a grid's levels, and both are measured
rather than asserted:

* **the ladder is learned** — `|w_i|` is a parameter, so it moves while the fit runs;
* **the bands are not spectral cuts** — every layer after the first mixes what it is
  given. Measured on the default fit, the share of a band tile's spectral energy that
  sits *above* that band's own top frequency is 70% at band 1, 90% at band 5, 93% at
  band 10 and 63% at band 15. That is why the montage renders every tile at the tile
  resolution instead of at the band's own Nyquist, and why "the band alone" is a
  marginal effect and not a term of a sum.

## Try it in the browser

Ports are ngp-demo's + 100, so both repos can run at once.

```bash
python scripts/gui_image.py     # http://localhost:8122  -- fit the painting
python scripts/gui_field.py     # http://localhost:8121  -- recover a known warp
python scripts/gui_time.py      # http://localhost:8124  -- a warp that moves
```

`gui_image.py` fits the painting and shows the frequency ladder the first layer is
working with, the effective band per pixel, and — live, while it trains — a 4x4
montage of what each band adds. **decompose** opens the montage full size.

`gui_field.py` runs the registration benchmark, a SIREN against a dense control grid,
scoring the field rather than the pixels. `gui_time.py` puts a slip band in motion
and fits one `(x, y, t)` SIREN to the whole run.

## omega_0 is the knob, and it fights the learning rate

400 steps, width 256 and 3 hidden layers, batch 262,144, on the 904x1069 painting.
PSNR in dB; 9–14 dB means the fit diverged and the panel is noise.

| omega_0 | lr 1e-4 | 5e-4 | 1e-3 | 5e-3 | finest first-layer wave |
|---|---|---|---|---|---|
| 30 | 27.94 | 29.01 | 28.83 | 14.27 | 237 px/cycle |
| 60 | 28.84 | **29.98** | 29.35 | 14.01 | 126 px/cycle |
| 120 | 29.26 | 29.02 | 14.28 | 14.01 | 65 px/cycle |
| 240 | 29.21 | 14.28 | 12.16 | 14.01 | 30 px/cycle |
| 480 | 29.00 | 11.45 | | | 17 px/cycle |
| 960 | 13.18 | 9.62 | | | 8.4 px/cycle |

The quality ridge is flat — every omega_0 from 30 to 480 lands within 2 dB of the
best, given a step it can survive — and the failures all lie on one diagonal.
**`omega_0 * lr` is what breaks:** every cell at 0.03 or below trains, every cell at
0.1 or above diverges. The page defaults to the best cell, omega_0 = 60 at lr 5e-4.

**The scales are not where the parameters are.** The default network holds 198,915
parameters, of which the first layer is **768** — 0.4%. Those 768 numbers fix every
starting frequency in the model; the other 99.6% decide what to do with them.

**The first layer does not reach the pixels, and does not need to.** At omega_0 = 60
its finest wave is 126 px per cycle on a 904 px image, and the fit still resolves the
eye and the pearl: the harmonics come from the layers above.

## One panel that had to be thrown away

ngp-demo draws, per 64 px block, *the finest level contributing there*. It is a real
question of a grid: a level only touches its own cells, so a block can name the level
that moved it. Ported to a SIREN it produced a uniform amber lattice, and the reason
is not a threshold to tune:

* every band is a plane wave across the whole frame. The busiest tenth of the blocks
  holds **18–33%** of any band's total (10% would be perfectly uniform);
* and every band contributes comparably — on the image fit, mean |Δ| runs 0.047 to
  0.073 across all sixteen; on the registration fit, 1.5 to 3.0 px.

So "the finest band clearing 8% of this block's peak" was satisfied by the top band
almost everywhere: **255 of 255 blocks** on the image page, **239 of 255** on the
registration page. True, and useless.

The panel now reports what is actually there: **what each band contributes and how
evenly** — bars of mean |Δ| on the band colour scale, with a white line for the
share sitting in the busiest tenth of the frame. On `gui_time.py`, where the question
is whether the network's use of frequency follows the moving slip band, the row shows
the **effective band per pixel** instead, and it does: a diagonal ridge at band ~9
against a background near 1, travelling with the band (frame means 3.81 / 3.40 / 3.68
over the three frames).

## Against the hash grid

Measured here and in ngp-demo at matched settings, same painting, same corrected
reference:

| | SIREN | hash grid |
|---|---|---|
| registration, slip_band, 400 steps | EPE fg **0.286 px**, 0 folded | (ngp-demo, 3000 steps: 0.324 px) |
| moving slip band, 200 frames, 400 steps | mean EPE **0.058 px**, 198,914 params | 0.116 px, 1,036,930 params |

The time-varying fit is the interesting one: half the endpoint error with a fifth of
the parameters. The grid's stage 2 needed its time axis capped at the frame spacing
or it memorised frames; a SIREN has no axis to cap — `t` is a third input to the same
plane waves — so that failure mode does not exist here.

The image fit goes the other way, and the page says so: at 400 steps the SIREN
reaches 29.98 dB where the hash grid reaches the mid 30s, because every weight here
touches the whole picture and the learning rate has to be two orders of magnitude
smaller.

## Install

Same environment as ngp-demo.

```bash
conda env create -f envs/environment.linux.yaml     # or environment.mac.yaml
conda activate ngp-demo
python tests/test_siren.py                          # 8 passed
python tests/check_pages.py                         # 4 pages, needs node
```
