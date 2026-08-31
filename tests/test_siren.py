#!/usr/bin/env python
"""Checks on the SIREN and on the frequency ladder this demo reads off it.

    python tests/test_siren.py            # all of them, ~20 s on a GPU

The ones that matter are the last three: the ladder is the whole premise of the
band montage and the coarse-to-fine window, so `frequency_i = omega_0 |w_i| / 2pi`
had better be the frequency the unit actually has, and gating a band had better
change exactly the units it claims.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siren import Siren, SirenField, jacobian, laplacian
from siren.siren import SineLayer

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_shapes_and_parameters():
    m = SirenField(n_input_dims=2, n_output_dims=3, width=64, hidden_layers=2)
    y = m(torch.rand(17, 2))
    assert y.shape == (17, 3), y.shape
    first, rest = m.n_parameters()
    # first layer: 64*2 weights + 64 bias
    assert first == 64 * 2 + 64, first
    assert first + rest == sum(p.numel() for p in m.parameters())
    assert (y >= 0).all() and (y <= 1).all(), "sigmoid output escaped [0, 1]"


@test
def test_init_follows_the_paper():
    """First layer U(-1/d, 1/d); hidden U(-sqrt(6/d)/w, +sqrt(6/d)/w)."""
    torch.manual_seed(0)
    w0, d, width = 30.0, 2, 4096
    m = SirenField(n_input_dims=d, width=width, hidden_layers=2, omega_0=w0)
    fw = m.first.linear.weight.detach()
    assert abs(float(fw.abs().max()) - 1 / d) < 0.02 * (1 / d), float(fw.abs().max())
    hid = m.net.net[1].linear.weight.detach()
    bound = np.sqrt(6 / width) / w0
    assert abs(float(hid.abs().max()) - bound) < 0.05 * bound, float(hid.abs().max())


@test
def test_gradcheck_wrt_coordinates():
    torch.manual_seed(0)
    m = SirenField(n_input_dims=2, n_output_dims=1, width=16, hidden_layers=1,
                   omega_0=5.0, output_activation="none").double()
    x = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q: m(q).sum(), (x,), eps=1e-6, atol=1e-6)


@test
def test_second_derivative_is_real():
    """A sine MLP has curvature.  A linearly interpolated grid does not, which is the
    reason ngp-demo needs smoothstep and this does not."""
    torch.manual_seed(0)
    m = SirenField(n_input_dims=2, n_output_dims=1, width=32, hidden_layers=2,
                   omega_0=20.0, output_activation="none")
    lap = laplacian(m, torch.rand(64, 2)).detach()
    assert lap.shape == (64, 1), lap.shape
    assert float(lap.abs().mean()) > 1e-3, float(lap.abs().mean())
    j = jacobian(m, torch.rand(8, 2)).detach()
    assert j.shape == (8, 1, 2), j.shape


@test
def test_frequency_matches_the_fft():
    """omega_0 |w_i| / 2pi against the peak of the unit's own spectrum.

    One unit, sampled along its own w_i, windowed, transformed, and the dominant
    bin read off.  If this disagrees the ladder is decoration.  omega_0 is large
    here on purpose: at the SIREN default of 30 a first-layer unit sits at 0.3 to
    2.7 cycles across the image, where one FFT bin IS the measurement and the test
    would be checking rounding rather than the formula.
    """
    torch.manual_seed(0)
    m = SirenField(n_input_dims=2, width=64, hidden_layers=1, omega_0=2000.0)
    f = m.frequencies()
    n = 8192
    t = torch.linspace(0, 1, n)
    win = torch.hann_window(n)
    worst = 0.0
    for i in torch.argsort(f)[[5, 20, 40, 60]]:                 # low to high
        w = m.first.linear.weight[i].detach()
        d = w / w.norm()
        xy = torch.stack([0.5 + (t - 0.5) * d[0], 0.5 + (t - 0.5) * d[1]], dim=1)
        with torch.no_grad():
            sig = torch.sin(m.omega() * (xy @ w + m.first.linear.bias[i]))
        spec = torch.fft.rfft((sig - sig.mean()) * win).abs()
        k = float(torch.argmax(spec))                # cycles over the sampled line
        worst = max(worst, abs(k - float(f[i])))
    assert worst < 0.6, f"frequency formula off by {worst:.2f} cycles"


@test
def test_band_window_gates_the_units_it_claims():
    torch.manual_seed(0)
    m = SirenField(n_input_dims=2, width=64, hidden_layers=1)
    band = m.band_of(16)
    assert band.min() == 0 and band.max() == 15
    counts = torch.bincount(band, minlength=16)
    assert int(counts.min()) == 4 and int(counts.max()) == 4, counts.tolist()
    f = m.frequencies()
    # bands are ordered in frequency
    means = torch.stack([f[band == b].mean() for b in range(16)])
    assert bool(torch.all(means[1:] > means[:-1])), means.tolist()
    m.set_band_window(4.0, 16)
    on = m.unit_gain > 0
    assert bool(torch.all(band[on] < 4)) and int(on.sum()) == 16, int(on.sum())


@test
def test_gating_a_band_changes_the_output():
    torch.manual_seed(0)
    m = SirenField(n_input_dims=2, n_output_dims=3, width=64, hidden_layers=2,
                   omega_0=40.0)
    x = torch.rand(256, 2)
    with torch.no_grad():
        full = m(x)
        m.set_band_window(8.0, 16)
        half = m(x)
        m.set_band_window(16.0, 16)
        back = m(x)
    assert float((full - half).abs().mean()) > 1e-3, "the window did nothing"
    assert torch.allclose(full, back, atol=1e-6), "the window did not restore"


@test
def test_vendored_siren_is_unchanged():
    """The vendored Siren still behaves like the one in connectome-gnn: sine layers,
    the /omega_0 hidden init, and an optional plain Linear on the output."""
    m = Siren(in_features=3, hidden_features=32, hidden_layers=2, out_features=1,
              outermost_linear=True, first_omega_0=80.0, hidden_omega_0=80.0)
    assert isinstance(m.net[0], SineLayer) and m.net[0].is_first
    assert isinstance(m.net[-1], torch.nn.Linear)
    assert m(torch.rand(5, 3)).shape == (5, 1)
    lm = Siren(in_features=1, hidden_features=8, hidden_layers=1, out_features=1,
               learnable_omega=True, first_omega_0=12.0)
    assert lm.get_omegas()[0] == 12.0
    assert float(lm.get_omega_L2_loss()) > 0


if __name__ == "__main__":
    torch.manual_seed(0)
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"ok  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(TESTS) - failed} passed" + (f", {failed} failed" if failed else ""))
    sys.exit(1 if failed else 0)
