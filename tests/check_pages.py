#!/usr/bin/env python
"""Load each GUI's page script in Node against a stub DOM and fail on any error.

    python tests/check_pages.py            # starts each server, checks, stops it

`node --check` only parses. It cannot see a `const` referenced before its
declaration, which is a ReferenceError that kills the entire script at load and
leaves every button dead and every panel black -- exactly the failure this
catches. The stub is deliberately dumb: enough DOM for the page to build its
controls and attach its handlers, and nothing more.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
# (script, port, [(path, expects a fit to start on load, feed it a populated
# state)]).  A server can serve more than one page: gui_image's decompose window
# is a second document with its own script, and an error in it is just as
# invisible as one in the main page.
PAGES = [
    ("scripts/gui_field.py", 8941, [("/", True, True)]),
    ("scripts/gui_image.py", 8942, [("/", True, True), ("/decompose", False, False)]),
    ("scripts/gui_time.py", 8943, [("/", True, True)]),
    ("scripts/gui_scalar_time.py", 8944, [("/", False, True)]),
]

# A realistic /api/state payload. The empty state exercises none of the drawing
# code, which is where a page actually breaks.
POPULATED = r"""
const PIX = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
          + "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const STATE = {
  running: true, step: 40, steps: 1500, seconds: 3.2, stamp: 7, note: "8..512 cells",
  images: {source: PIX, target: PIX, warped: PIX, epe: PIX, reference: PIX,
           fit: PIX, error: PIX, levels: PIX},
  grid: {w: 904, h: 1069, epe_vmax: 3,
         gt: [[[0,0],[100,0],[200,10]], [[0,0],[0,100]]],
         fit: [[[0,1],[100,2],[200,12]], [[1,0],[1,100]]]},
  curve: [{step: 0, loss: 1e-2, epe_fg: 2.0, epe_bg: 3.0, t: 0.0, psnr: 20},
          {step: 40, loss: 1e-3, epe_fg: 0.5, epe_bg: 1.0, t: 3.2, psnr: 35}],
  metrics: {psnr: 35.5, loss: 1e-3, loss_kind: "l2", n_parameters: 123456,
            epe_fg: 0.5, epe_band: 0.8, epe_bg: 1.0, det_min: 0.4,
            folded: 0, folded_count: 0, jacobian_samples: 107156,
            n_first: 768, n_rest: 198147, n_total: 198915, n_values: 2899128,
            fraction_of_values: 0.42, n_bands: 16, omega: 60,
            f_min: 0.4, f_max: 7.2, finest_px_per_cycle: 126.0,
            outermost_linear: true, learnable: false,
            width: 904, height: 1069, channels: 3},
  ladder: [{band: 0, units: 16, f_lo: 0.4, f_hi: 0.9, px: 1400.0},
           {band: 15, units: 16, f_lo: 6.1, f_hi: 7.2, px: 136.0}],
  blocks: {n_levels: 16, w: 904, h: 1069,
           profile: [{band: 0, mean: 0.031, top10: 0.18},
                     {band: 15, mean: 0.009, top10: 0.23}],
           blocks: [{x: 0, y: 0, w: 64, h: 64, level: 3, cell_px: 335.0}]},
  history: [{label: "L15", params: 123456, psnr: 35.5, seconds: 3.2,
             curve: [{t: 0, psnr: 20}, {t: 3.2, psnr: 35}]}],
  info: {n_first: 768, n_rest: 198147, n_total: 198915, n_values: 2899128,
         fraction_of_values: 0.42, n_bands: 16, omega: 60, f_min: 0.4,
         f_max: 7.2, finest_px_per_cycle: 126.0, outermost_linear: true,
         learnable: false, width: 904, height: 1069, channels: 3},
  pyramid_sigma: 8,
};
global.fetch = () => Promise.resolve({json: () => Promise.resolve(STATE)});
poll().then(() => console.log("  populated poll ok"))
      .catch(e => { console.error("populated poll threw:", e && e.stack || e);
                    process.exitCode = 1; });
"""

STUB = r"""
// Minimal DOM: every element answers every call, so the page can build itself.
const noop = () => {};
function ctx() {
  return new Proxy({}, {get: (t, k) => {
    if (k === "canvas") return {width: 330, height: 460};
    if (["fillStyle","strokeStyle","lineWidth","font","globalAlpha",
         "imageSmoothingEnabled","textAlign"].includes(k)) return "";
    return noop;
  }, set: () => true});
}
function el(id) {
  const e = {
    id: id || "", innerHTML: "", textContent: "", value: 0, type: "",
    width: 330, height: 460, style: {},
    children: [], className: "", checked: false, dataset: {},
    classList: {add: noop, remove: noop, toggle: noop, contains: () => false},
    appendChild: c => { e.children.push(c); return c; },
    append: (...c) => { e.children.push(...c); },
    setAttribute: noop, getAttribute: () => null, addEventListener: noop,
    removeEventListener: noop, getContext: ctx, focus: noop,
    click: () => { if (typeof e.onclick === "function") e.onclick({}); },
    getBoundingClientRect: () => ({left: 0, top: 0, width: 330, height: 460}),
  };
  return e;
}
const ELS = {};
global.document = {
  getElementById: id => (ELS[id] = ELS[id] || el(id)),
  createElement: tag => el(""),
  addEventListener: noop, querySelector: () => el(""),
  querySelectorAll: () => [],
};
global.window = global;
global.addEventListener = noop;      // the page registers unhandledrejection on window
// Fire onload. Without it the stub never runs blit(), which is the function
// that actually paints -- so a page could pass the check and still show four
// black panels in a browser.
global.Image = class {
  constructor() { this.width = 389; this.height = 460; this._src = ""; }
  set src(v) { this._src = v; if (this.onload) this.onload(); }
  get src() { return this._src; }
};
global.CALLS = [];
global.fetch = (u) => { global.CALLS.push(String(u));
  return Promise.resolve({json: () => Promise.resolve({
  running: false, step: 0, steps: 0, seconds: 0, curve: [], metrics: {},
  images: {}, grid: {}, blocks: {}, ladder: [], history: [], note: "",
  stamp: 0, info: {}, levels: {}, frames: [0, 1, 2], per_frame: [],
  panels: [], n_levels: 18, render: "301x356"})}); };
global.setInterval = () => 0;
global.clearInterval = noop;
global.setTimeout = (f) => 0;
global.clearTimeout = noop;
global.requestAnimationFrame = () => 0;
process.on("exit", () => {
  if (global.EXPECT_START && !global.CALLS.some(u => u.includes("/api/start"))) {
    console.error("page never called /api/start on load");
    process.exitCode = 1;
  }
});
"""


def check(script: str, port: int, paths) -> bool:
    proc = subprocess.Popen([PY, "-u", os.path.join(ROOT, script), "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(40):
            time.sleep(1.0)
            if proc.poll() is not None:
                print(f"  {script}: server exited: "
                      f"{proc.stderr.read().decode()[-300:]}")
                return False
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read()
                break
            except Exception:
                continue
        else:
            print(f"  {script}: server never answered")
            return False
        return all(check_page(script, port, *p) for p in paths)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def check_page(script, port, path, expect_start, populated) -> bool:
        name = f"{script}{'' if path == '/' else ' ' + path}"
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                      timeout=10).read().decode()
        left = [t for t in re.findall(r"__[A-Z_]+__", html)]
        if left:
            print(f"  {name}: unreplaced placeholders {sorted(set(left))}")
            return False
        # The explainer is a modal. Without the rule that hides it, it renders
        # inline and pushes the panels off the page -- which looks exactly like
        # "clicking run does nothing".
        if 'class="modal"' in html and ".modal { position:fixed" not in html:
            print(f"  {name}: modal markup present but the stylesheet that "
                  "hides it is not")
            return False
        js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        out = f"/tmp/_page_{port}{path.replace('/', '_')}.js"
        with open(out, "w") as f:
            # every main page opens on a running fit; assert that it really does
            f.write(STUB
                    + (f"\nglobal.EXPECT_START = {str(expect_start).lower()};\n")
                    + js + ("\n" + POPULATED if populated else ""))
        r = subprocess.run(["node", out], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  {name}: page script threw\n"
                  + "\n".join(r.stderr.strip().splitlines()[:12]))
            return False
        print(f"  {name}: page builds cleanly ({len(js):,} chars of script)")
        return True


if __name__ == "__main__":
    ok = all([check(s, p, paths) for s, p, paths in PAGES])
    print("\nall pages ok" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
