"""Shared pieces of the two browser UIs: image encoding and the flat stylesheet.

Both `scripts/gui_field.py` (registration) and `scripts/gui_image.py` (image fitting)
serve a single self-contained page and poll a JSON endpoint, so the only things
worth sharing are how a tensor becomes a PNG data URI and what the page looks
like.
"""

from __future__ import annotations

import base64
import io

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
from PIL import Image

DISPLAY_H = 460                      # panel height in px; images are sent downsampled


def png_data_uri(rgb: np.ndarray, max_h: int = DISPLAY_H) -> str:
    im = Image.fromarray(rgb)
    if im.height > max_h:
        im = im.resize((max(1, round(im.width * max_h / im.height)), max_h),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_png(t: torch.Tensor, max_h: int = DISPLAY_H) -> str:
    a = np.clip(t.detach().cpu().numpy(), 0, 1)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return png_data_uri((a * 255).astype(np.uint8), max_h)


# The band ramp, shared with the pages' levColor(): bright at both ends, so the
# low bands are visible against a dark image and the legend means the same thing
# in the overlay and in the map.
LEVEL_RAMP = LinearSegmentedColormap.from_list(
    "levels", [(0.30, 0.64, 1.00), (0.25, 0.88, 0.82), (0.49, 1.00, 0.35),
               (1.00, 0.82, 0.30), (1.00, 0.42, 0.42)])


# Diverging, through black rather than white: these panels sit on a black page,
# so zero should read as "nothing here" and not as the brightest thing in the
# frame.  Blue negative, red positive, symmetric about zero.
SIGNED_RAMP = LinearSegmentedColormap.from_list(
    "signed", [(0.25, 0.55, 1.00), (0.10, 0.20, 0.45), (0.00, 0.00, 0.00),
               (0.45, 0.12, 0.12), (1.00, 0.30, 0.28)])


def signed_rgb(a: np.ndarray, vmax: float) -> np.ndarray:
    """Signed data on a symmetric +-vmax scale -> uint8 RGB."""
    x = np.clip(a / max(vmax, 1e-6), -1.0, 1.0) * 0.5 + 0.5
    return (SIGNED_RAMP(x)[..., :3] * 255).astype(np.uint8)


def signed_png(a: np.ndarray, vmax: float, max_h: int = DISPLAY_H) -> str:
    return png_data_uri(signed_rgb(a, vmax), max_h)


def cmap_png(a: np.ndarray, vmax: float, name="inferno", max_h: int = DISPLAY_H) -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    cmap = LEVEL_RAMP if name == "levels" else matplotlib.colormaps[name]
    return png_data_uri((cmap(x)[..., :3] * 255).astype(np.uint8), max_h)


def field_png(a: np.ndarray, vmax: float, name="viridis",
              max_h: int = DISPLAY_H) -> str:
    """A SIGNED scalar field on a symmetric [-vmax, vmax], through viridis.

    Symmetric because zero is a meaningful value in a wave and should land in the
    same colour whatever the frame's own extremes are, and fixed because a panel
    that rescales itself each refresh cannot be compared with the one beside it.
    """
    x = np.clip(a / max(vmax, 1e-6), -1.0, 1.0) * 0.5 + 0.5
    return png_data_uri((matplotlib.colormaps[name](x)[..., :3] * 255).astype(np.uint8),
                        max_h)


def flow_png(u: np.ndarray, vmax: float, max_h: int = DISPLAY_H) -> str:
    """A displacement field as the optical-flow colour wheel: hue is direction,
    brightness is magnitude.

    A magnitude map alone is blind to the most interesting cases. A shear band
    displaces two half-planes by equal and opposite amounts, so |u| is uniform
    and the map is flat -- the thing that makes it a shear is entirely in the
    direction, which this shows as two opposed hues meeting at a line.

    u: (H, W, 2) in pixels.
    """
    import colorsys
    ang = (np.arctan2(u[..., 1], u[..., 0]) / (2 * np.pi)) % 1.0
    mag = np.clip(np.linalg.norm(u, axis=-1) / max(vmax, 1e-6), 0, 1)
    hsv = np.stack([ang, np.ones_like(mag), mag], -1).reshape(-1, 3)
    rgb = np.array([colorsys.hsv_to_rgb(*c) for c in hsv]).reshape(*u.shape[:2], 3)
    return png_data_uri((rgb * 255).astype(np.uint8), max_h)


CSS = """
  :root { --fg:#fff; --bg:#000; --dim:#9a9a9a; --red:#e5484d; --blue:#4da3ff;
          --amber:#e5a23c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1400px; margin:0 auto; padding:26px 22px 48px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 6px; }
  .sub { font-size:12px; color:var(--dim); margin:0 0 22px; max-width:960px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:18px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .seg button:disabled { opacity:.35; cursor:default; }
  .knobs { display:flex; flex-wrap:wrap; gap:26px; margin:0 0 14px;
           padding:14px 16px; border:1px solid #333; }
  .knobs .title { width:100%; font-size:10px; letter-spacing:.16em;
                  text-transform:uppercase; color:var(--dim); margin-bottom:2px; }
  .knob { display:flex; flex-direction:column; gap:5px; min-width:250px; flex:1; }
  .knob .kl { font-size:11px; color:var(--dim); display:flex;
              justify-content:space-between; gap:12px; }
  .knob .kl b { color:var(--fg); font-weight:600;
                font-variant-numeric:tabular-nums; }
  .knob .ends { display:flex; justify-content:space-between; font-size:9px;
                color:#666; font-variant-numeric:tabular-nums; }
  input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                      height:1px; background:var(--fg); outline:none; margin:6px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
    appearance:none; width:13px; height:13px; background:var(--fg);
    border:1px solid var(--fg); cursor:pointer; border-radius:0; }
  input[type=range]::-moz-range-thumb { width:13px; height:13px;
    background:var(--fg); border:1px solid var(--fg); cursor:pointer;
    border-radius:0; }
  .row { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid #333; }
  /* A row of equal panels that must not wrap: the canvases keep their pixel
     backing store and are scaled by CSS, so four of them share the width
     evenly however narrow the window is. */
  .row.equal { flex-wrap:nowrap; }
  .row.equal .panel { flex:1 1 0; min-width:0; }
  .row.equal canvas { width:100%; height:auto; }
  /* Four columns on the same track as a four-panel .row.equal above it: same
     gap, same 1fr columns, so a panel here lines up with the panel above it
     however wide the window is.  A flex row cannot do this -- its free space is
     split after removing ITS OWN gaps, so a row of three lands ~9 px off. */
  .row.grid4 { display:grid; grid-template-columns:repeat(4, 1fr); gap:18px;
               align-items:start; }
  .row.grid4 .panel { min-width:0; }
  .row.grid4 .span2 { grid-column:span 2; }
  .row.grid4 canvas { width:100%; height:auto; }
  .cap { font-size:10px; letter-spacing:.14em; text-transform:uppercase;
         color:var(--dim); }
  .cap i { color:var(--red); font-style:normal; }
  .cap b { color:var(--blue); font-weight:600; }
  .stats { font-size:12px; color:var(--dim); margin-top:18px;
           font-variant-numeric:tabular-nums; line-height:1.9; }
  .stats b { color:var(--fg); font-weight:600; }
  .stats .bad { color:var(--red); }
  .bar { height:2px; background:#222; margin:14px 0 0; }
  .bar i { display:block; height:2px; background:var(--fg); width:0; }
  .note { font-size:11px; color:#7a7a7a; margin-top:6px; }
  .setup { font-size:12px; color:#fff; margin:12px 0 2px;
           font-variant-numeric:tabular-nums; }
  .setup b { font-weight:600; }
  .setup span.dim { color:var(--dim); }
  table.ladder { border-collapse:collapse; font-size:11px;
                 font-variant-numeric:tabular-nums; }
  table.ladder th { text-align:right; font-weight:600; color:var(--dim);
                    padding:2px 10px; font-size:10px; letter-spacing:.1em;
                    text-transform:uppercase; }
  table.ladder td { text-align:right; padding:2px 10px; color:#d8d8d8; }
  table.ladder tr.hashed td { color:var(--amber); }
  .modal { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none;
           z-index:50; overflow:auto; }
  .modal.open { display:block; }
  .modal .sheet { max-width:820px; margin:6vh auto; background:#000;
                  border:1px solid var(--fg); padding:28px 30px 34px; }
  .modal h2 { font-size:13px; letter-spacing:.14em; text-transform:uppercase;
              margin:22px 0 8px; font-weight:600; }
  .modal h2:first-of-type { margin-top:0; }
  .modal p { font-size:13px; line-height:1.6; color:#d0d0d0; margin:0 0 10px; }
  .modal code { color:var(--amber); font-family:ui-monospace,Menlo,monospace;
                font-size:12px; }
  .modal a { color:var(--blue); }
  .modal .close { float:right; background:#000; color:var(--fg);
                  border:1px solid var(--fg); padding:5px 12px; cursor:pointer;
                  font:inherit; font-size:12px; }
  .modal ul { margin:0 0 10px; padding-left:20px; color:#d0d0d0; font-size:13px;
              line-height:1.6; }
"""

# One explainer, shared by both pages: what the encoding is, what each control
# changes, and which claims here were measured rather than assumed.


# Both explainers run coarse to specific: the headline, then the mechanism, then
# the detail. A reader who stops after the first paragraph should still have the
# true shape of it.

ABOUT_HTML = """
<button class="close" onclick="closeAbout()">close</button>

<h2>In one sentence</h2>
<p>A coordinate goes in and a value comes out &mdash; a colour, a density, a
displacement &mdash; through an ordinary MLP whose activation is
<code>sin</code> instead of a ReLU, and that single substitution is the whole
method.</p>

<h2>The mechanism</h2>
<p>Every layer computes <code>sin(omega * (W x + b))</code>. There is no grid, no
table and no lookup: the network is the representation, and the detail lives in the
frequencies its sines can reach.</p>
<p>Two details make it work rather than merely run. The <b>init</b> is chosen so the
pre-activations keep a fixed distribution however deep the stack goes: the first
layer draws <code>W ~ U(-1/d, 1/d)</code>, every later layer draws
<code>W ~ U(-sqrt(6/d)/omega, +sqrt(6/d)/omega)</code>, so the <code>omega</code>
multiplying the layer is cancelled by the weights it multiplies. And
<b><code>omega_0</code></b>, the first layer's factor, sets the scale the network
starts from.</p>
<p>Sitzmann, Martel, Bergman, Lindell and Wetzstein, <i>Implicit Neural
Representations with Periodic Activation Functions</i>, NeurIPS 2020
(<a href="https://arxiv.org/abs/2006.09661" target="_blank">arXiv:2006.09661</a>,
<a href="https://github.com/vsitzmann/siren" target="_blank">vsitzmann/siren</a>).
The implementation here is the one already in this workspace, in
<code>connectome-gnn</code>, where it learns the visual stimulus.</p>

<h2>The first layer is a bank of plane waves</h2>
<p>This is what the pages here are built on. Unit <code>i</code> of the first layer
computes <code>sin(omega_0 (w_i . x + b_i))</code>, and along the direction of
<code>w_i</code> that is a wave of</p>
<p><code>f_i = omega_0 |w_i| / 2pi</code> &nbsp; cycles across the domain.</p>
<p>So the first layer <i>has</i> a ladder of scales, one rung per unit, and it can be
read off the weights. Sorted by <code>f_i</code> and cut into sixteen equal-count
bands, that ladder is the direct analogue of a hash grid's levels: it can be drawn,
switched off band by band, released coarse-to-fine, and taken apart into a
montage.</p>
<p>Two things are true of it that are not true of a grid's levels.</p>
<ul>
<li><b>It is learned.</b> <code>|w_i|</code> is a parameter, so the ladder moves while
the fit runs. The ladder panel is a measurement, not a setting.</li>
<li><b>It is only the start.</b> A sine of a sine generates harmonics, so the layers
above manufacture frequencies the first layer never had. At the default
<code>omega_0 = 60</code> on this 904 px painting the first layer's finest wave is
<b>126 px per cycle</b> &mdash; nothing like pixel detail &mdash; and the fit still
resolves the eye and the pearl. Gating a band therefore changes what every later
layer sees; a band tile is the marginal effect of releasing those units, not a
spectral component of the output.</li>
</ul>

<h2>omega_0 is the knob, and it fights the learning rate</h2>
<p>Measured on this painting, 400 steps, width 256 and 3 hidden layers, PSNR in dB
(9&ndash;14 dB means the fit diverged and the panel is noise):</p>
<table class="ladder"><tr><th>omega_0</th><th>lr 1e-4</th><th>5e-4</th><th>1e-3</th>
<th>5e-3</th><th>finest first-layer wave</th></tr>
<tr><td>30</td><td>27.94</td><td>29.01</td><td>28.83</td><td>14.27</td>
<td>237 px/cycle</td></tr>
<tr><td>60</td><td>28.84</td><td><b>29.98</b></td><td>29.35</td><td>14.01</td>
<td>126 px/cycle</td></tr>
<tr><td>120</td><td>29.26</td><td>29.02</td><td>14.28</td><td>14.01</td>
<td>65 px/cycle</td></tr>
<tr><td>240</td><td>29.21</td><td>14.28</td><td>12.16</td><td>14.01</td>
<td>30 px/cycle</td></tr>
<tr><td>480</td><td>29.00</td><td>11.45</td><td></td><td></td>
<td>17 px/cycle</td></tr>
<tr><td>960</td><td>13.18</td><td>9.62</td><td></td><td></td>
<td>8.4 px/cycle</td></tr>
</table>
<p>The quality ridge is flat &mdash; everything from 30 to 480 lands within 2 dB of the
best, given a learning rate it can survive &mdash; and the failures are all on one
diagonal. <b>The product <code>omega_0 * lr</code> is what breaks:</b> every cell at
0.03 or below trains, every cell at 0.1 or above diverges. That is the same
statement as "the first layer's gradient is scaled by <code>omega_0</code>", and it
is why raising <code>omega_0</code> without lowering the step turns the fit to
noise.</p>

<h2>Where the parameters are, and where the scales are</h2>
<p>They are not the same place. The default network here holds <b>198,915</b>
parameters, of which the first layer is <b>768</b> &mdash; 0.4%. Those 768 numbers fix
every starting frequency in the model; the other 99.6% decide what to do with
them.</p>

<h2>What this costs against a hash grid</h2>
<p>Every parameter here touches the whole picture, which is the opposite of a grid,
where an entry touches one cell. That is why the learning rate is two orders of
magnitude smaller (5e-4 against 1e-2), why there is no notion of "empty space is
free", and why the derivatives are worth having: a sine MLP is smooth, so
<code>d2f/dx2</code> is real, where a linearly interpolated grid has an identically
zero second derivative.</p>
"""


_TRAINING = """
<h2>Training</h2>
<p>Three knobs, and they behave as they do anywhere else.</p>
<ul>
<li><b>learning rate</b> &mdash; Adam's step, log-spaced, cosine-decayed to 3% of its
starting value. A SIREN wants roughly 5e-4, two orders of magnitude under a hash
table's 1e-2, and the ceiling moves with <code>omega_0</code>: measured here, the fit
survives <code>omega_0 * lr</code> up to about 0.03 and diverges by 0.1.</li>
<li><b>iterations</b> &mdash; how long. Every schedule on the page is expressed as a
fraction of this, so changing it rescales them rather than truncating them.</li>
<li><b>batch size</b> &mdash; sample points per step. Uniform over the image on the
fitting page; on the registration page, 90% inside the foreground mask and 10%
uniform.</li>
</ul>
"""

_TERMINAL = """
<h2>If the page looks wrong</h2>
<p>The terminal prints <code>[run]</code> with the configuration,
<code>[images]</code> when the first frames go out, and <code>[done]</code> or
<code>[stopped]</code> with the final numbers. If those lines are there and the page
is blank, the fault is in the browser and not in the fit; the page also reports its
own exceptions back to the terminal as <code>[client]</code>.</p>
"""

INTERFACE_IMAGE = """
<button class="close" onclick="closeHelp()">close</button>

<h2>What this page does</h2>
<p>It fits the painting &mdash; random pixel coordinates in, RGB out &mdash; with a
SIREN, and shows the ladder of scales its first layer is working with. The number in
the line above the panels is the compression figure: parameters against the image's
own value count, green under 50%, amber under 100%, red once your "compression" is an
expansion.</p>

<h2>The three knobs</h2>
<ul>
<li><b>width</b> and <b>hidden sine layers</b> &mdash; the MLP. Width is also the number
of plane waves the first layer starts from, so it sets how finely the ladder can be
cut: 256 units across 16 bands is 16 units per band.</li>
<li><b>omega_0</b> &mdash; the first layer's frequency factor, and the knob that
matters. Unit <code>i</code> starts at <code>omega_0 |w_i| / 2pi</code> cycles across
the picture, so doubling it doubles every starting wave. Log-spaced, because the
useful range spans two decades. It fights the learning rate: see <i>what is a
siren?</i> for the measured table, but the rule is that
<code>omega_0 * lr</code> above ~0.1 diverges.</li>
</ul>

<h2>Two switches, which change what the network is</h2>
<ul>
<li><b>output layer</b> &mdash; a plain <code>Linear</code> on the output, or one more
sine. Linear is the reference implementation's default for images and is what
<code>outermost_linear=True</code> means there.</li>
<li><b>omega_0 fixed / learnable</b> &mdash; the copy of this network in
<code>connectome-gnn</code> added a trainable omega, with an L2 to keep it from
running away. Turning it on lets the fit choose its own scale, and the ladder panel
shows what it chose.</li>
<li><b>downsample</b> &mdash; 1, 2 or 4. It changes the reference the compression figure
is measured against, so the same network reads four times larger at downsample 2.</li>
</ul>
""" + _TRAINING + """
<h2>Reading the panels</h2>
<ul>
<li><b>reference / fit / absolute error</b> &mdash; error on a fixed 0-0.1 scale, so
it darkens as the fit improves rather than rescaling itself.</li>
<li><b>finest band contributing</b> &mdash; the image is divided into fixed 64 px
<i>analysis blocks</i>; each is coloured by the finest band clearing 8% of that
block's strongest contribution, and drawn with that band's <i>wavelength</i> as a
lattice &mdash; one square per cycle &mdash; if it is at least 3 screen pixels. Blocks
whose band is finer than that are tinted instead, so lower omega_0 to see real
cycles.</li>
<li><b>the 16 frequency bands</b> &mdash; the network taken apart while it trains. By
default each tile is what that band <i>adds</i>, signed: blue negative, black zero,
red positive, on a fixed &plusmn;0.1. The first tile is the baseline the differences
start from, and it is not black &mdash; with every unit gated off the first layer emits
zeros and the layers above return a constant. Baseline plus the differences is the
fit. Each tile is sampled at two points per cycle of its own band and blown up with
nearest-neighbour, so a band of six cycles reads as twelve samples rather than as a
smooth blur the display invented. <b>decompose</b> opens the same thing full size,
with a view that puts one band through the rest of the network alone.</li>
<li><b>psnr against training time</b> &mdash; finished runs stay, colour-keyed to the
table beside them, so settings compare on quality against time and parameters.</li>
<li><b>magnifier</b> &mdash; hover to magnify; with <i>reference fixed</i> the first
panel stays whole and marks the region the others show. Scroll changes the
factor.</li>
</ul>
""" + _TERMINAL

INTERFACE_REG = """
<button class="close" onclick="closeHelp()">close</button>

<h2>What this page does</h2>
<p>It warps the painting by a <i>known</i> analytic field, hands the pair to a
parameterisation, and scores the <b>field</b> it recovers rather than the pixels it
matches. Because the truth is analytic, the score is the <b>endpoint error</b> &mdash;
the distance in pixels between the recovered displacement and the true one &mdash; split
three ways: the textured foreground, the black background where nothing constrains
the warp, and the band between them.</p>

<h2>The two parameterisations</h2>
<ul>
<li><b>siren</b> &mdash; an MLP of sines, <code>(x, y) -&gt; u</code>. It has no grid, so
its scales are the first layer's plane waves: unit <code>i</code> starts at
<code>omega_0 |w_i| / 2pi</code> cycles across the frame.</li>
<li><b>tensor_16 / 64 / 256</b> &mdash; a dense control tensor, bilinearly interpolated.
Structurally smooth, which is a prior and not a defect: where there is no data, the
prior is the whole answer.</li>
</ul>

<h2>The knobs that are the model</h2>
<ul>
<li><b>width</b> and <b>hidden sine layers</b> &mdash; the network. Width is also how many
plane waves the first layer starts from.</li>
<li><b>omega_0</b> &mdash; the scale the first layer starts at, set against the
<i>deformation</i> and not the picture. The finest thing in these warps is a 12 px
shear band on a 904 px frame, so about 75 cycles; omega_0 = 30 starts the first layer
well under that and lets the layers above build the rest.</li>
<li><b>control points per axis</b> &mdash; the control grid's only knob.</li>
</ul>

<h2>Getting an intensity loss to converge</h2>
<ul>
<li><b>image pyramid</b> &mdash; blurred copies of both images, coarse first. An
intensity loss can only see a displacement smaller than the structure it is
comparing, so without a pyramid a 40 px ground-truth shift is invisible to a 9 px
patch and the fit sits in the nearest local minimum.</li>
<li><b>coarse to fine (band window)</b> &mdash; the same idea applied to the model
instead of the data. It starts with 4 of the 16 frequency bands live and ramps to all
of them by half-way, gating the first layer's units by
<code>clamp(alpha - band, 0, 1)</code>. This is the direct analogue of the hash grid's
level window, and it is the same gain vector the band montage uses.</li>
</ul>

<h2>Reading the panels</h2>
<ul>
<li><b>finest band contributing</b> &mdash; 64 px analysis blocks coloured by the
finest level doing work in each. Signal versus none, not local scale.</li>
<li><b>grid</b> &mdash; a regular grid carried through the warp, ground truth in red
against the fit in blue dashes. Where they coincide the field is right, which is a
stronger statement than the images matching.</li>
<li><b>endpoint error</b> &mdash; fixed 0-10 px scale.</li>
<li><b>the curve</b> &mdash; endpoint error by region. The <b>background</b> line
sits near the ground-truth displacement magnitude and stays there: nothing constrains
the warp where there is no image content, so it reports how much warp exists out of
reach rather than how good the fit is.</li>
</ul>
""" + _TERMINAL
