#!/usr/bin/env python
"""Merge the benchmark's result files into one table and one figure.

    python scripts/summarise.py

The runs land in several directories because they come from several invocations
(the main grid, the coarse-to-fine ablation, the encoding arms).  A row is keyed
by (deformation, mismatch, model); when the same key appears twice, the later
directory in `--dirs` wins, which is how a corrected re-run supersedes an
earlier one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLS = [("model", "model", "{}"),
        ("psnr_warped_vs_clean_target", "PSNR", "{:.2f}"),
        ("endpoint_error_px_foreground", "EPE fg", "{:.3f}"),
        ("endpoint_error_px_boundary_band", "EPE band", "{:.3f}"),
        ("endpoint_error_px_background", "EPE bg", "{:.3f}"),
        ("jacobian_det_min", "min det J", "{:.3f}"),
        ("jacobian_det_negative_fraction", "folded", "{:.4f}"),
        ("bending_energy", "bending", "{:.2e}"),
        ("n_parameters", "params", "{:,}"),
        ("train_seconds", "s", "{:.0f}")]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="*", default=[
        os.path.join(ROOT, "out/registration"),
        os.path.join(ROOT, "out/registration_c2f"),
        os.path.join(ROOT, "out/registration_enc"),
        os.path.join(ROOT, "out/registration_c2f_nopyr")])
    p.add_argument("--out", default=os.path.join(ROOT, "out/registration_summary"))
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    rows = {}
    for d in args.dirs:
        f = os.path.join(d, "results.json")
        if not os.path.exists(f):
            print(f"  (missing {f})")
            continue
        with open(f) as fh:
            got = json.load(fh)
        for r in got:
            # The pyramid belongs in the key: a no-pyramid run is a different
            # configuration, not a correction of the same one, and keying
            # without it made the two silently overwrite each other.
            key = (r["deformation"], r["mismatch"], r["model"],
                   r.get("pyramid", True))
            if key in rows:
                print(f"  superseded by {os.path.basename(d)}: {key}")
            rows[key] = r
        print(f"  {len(got):3d} rows from {os.path.basename(d)}")
    if not rows:
        sys.exit("no results found -- run scripts/run_registration.py first")

    ordered = sorted(rows.values(),
                     key=lambda r: (r["deformation"], r["mismatch"],
                                    not r.get("pyramid", True), r["model"]))
    lines = ["# Registration benchmark", "",
             f"{len(ordered)} runs. EPE = endpoint error against the analytic field, "
             "in pixels: `fg` is the textured foreground, `bg` the zeroed background "
             "where nothing constrains the warp, `band` the 24 px boundary between "
             "them. In the background every model predicts nearly zero displacement, "
             "so EPE there mostly reports how much ground-truth warp exists in a "
             "region no data can reach.", ""]
    for key in dict.fromkeys((r["deformation"], r["mismatch"],
                              r.get("pyramid", True)) for r in ordered):
        sel = [r for r in ordered
               if (r["deformation"], r["mismatch"], r.get("pyramid", True)) == key]
        loss = sel[0].get("loss", "l2")
        pyr = "" if key[2] else ", no image pyramid"
        lines += [f"### {key[0]} / {key[1]}  ({loss} loss{pyr})", "",
                  "| " + " | ".join(c[1] for c in COLS) + " |",
                  "|" + "---|" * len(COLS)]
        for r in sel:
            lines.append("| " + " | ".join(
                c[2].format(r[c[0]]) if c[0] in r else "-" for c in COLS) + " |")
        lines.append("")
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(ordered, f, indent=1)
    print("\n".join(lines))

    figure(ordered, args.out)
    print(f"wrote {args.out}")


def figure(rows, out):
    """Endpoint error by region, one panel per (deformation, mismatch)."""
    keys = list(dict.fromkeys((r["deformation"], r["mismatch"],
                               r.get("pyramid", True)) for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.4 * nrow), squeeze=False)
    regions = [("endpoint_error_px_foreground", "foreground", "tab:blue"),
               ("endpoint_error_px_boundary_band", "band", "tab:purple"),
               ("endpoint_error_px_background", "background", "tab:orange")]
    for i, key in enumerate(keys):
        a = ax[i // ncol][i % ncol]
        sel = {r["model"]: r for r in rows
               if (r["deformation"], r["mismatch"], r.get("pyramid", True)) == key}
        present = [m for m in models if m in sel]
        x = np.arange(len(present))
        for j, (field, lab, col) in enumerate(regions):
            v = [max(sel[m][field], 1e-3) for m in present]
            a.bar(x + (j - 1) * 0.27, v, width=0.26, color=col,
                  label=lab if i == 0 else None)
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
        a.set_ylabel("endpoint error (px)", fontsize=11)
        a.grid(alpha=0.25, axis="y")
        a.text(0.02, 0.98, "abcdefghijkl"[i], transform=a.transAxes, va="top",
               ha="left", fontsize=15, fontweight="bold")
        a.text(0.98, 0.98, f"{key[0]}\n{key[1]}" + ("" if key[2] else "\nno pyramid"),
               transform=a.transAxes,
               va="top", ha="right", fontsize=10, color="0.35")
    for i in range(len(keys), nrow * ncol):
        ax[i // ncol][i % ncol].axis("off")
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "epe_by_region.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
