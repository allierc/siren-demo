"""SIREN (Sitzmann et al. 2020), with the frequency machinery this demo needs.

`SineLayer` and `Siren` are vendored from
`connectome-gnn/src/connectome_gnn/models/Siren_Network.py`, which adapted them from
https://github.com/vsitzmann/siren.  They are kept as they are there -- same init,
same `outermost_linear`, same `learnable_omega` -- so a fit here and a fit in that
repo are the same network.

What is added is `SirenField`, and one idea:

    the first layer is a bank of plane waves.

Unit i of the first layer computes sin(w0 * (w_i . x + b_i)), which on x in [0, 1]^d
is a plane wave of  w0 * |w_i| / 2pi  cycles across the domain.  Sorting the units by
that number gives a SIREN something a hash grid has by construction and a SIREN is
usually said to lack: a ladder of scales, from a few cycles across the picture to a
few pixels per cycle.  Gating those units with a per-unit gain vector is then the
exact analogue of the hash grid's `level_gain`, and everything ngp-demo builds on
`level_gain` -- the band montage, the effective-scale map, the coarse-to-fine window
-- transfers unchanged.

Two things that are true of this ladder and not of the grid's:

  * it is LEARNED.  |w_i| moves during training, so the ladder is a readout of the
    fit rather than a setting of it.
  * deeper layers mix.  Gating a band changes what every later layer sees, so a band
    tile is the marginal effect of releasing those units, not a spectral component of
    the output.  tests/band_specialisation.py measures how far apart those two are.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """sin(omega_0 * (W x + b)).

    See the SIREN paper, sec. 3.2 final paragraph and supplement sec. 1.5.  If
    `is_first`, omega_0 multiplies the activations before the nonlinearity and is a
    hyperparameter of the signal.  Otherwise the weights are divided by omega_0 so the
    activation magnitude stays put while the gradients to the weights are boosted.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, learnable_omega=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features
        self.learnable_omega = learnable_omega

        if learnable_omega:
            self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        else:
            self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(omega_0)

    def init_weights(self, omega_0):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega_0,
                                            np.sqrt(6 / self.in_features) / omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.,
                 learnable_omega=False):
        super().__init__()

        self.learnable_omega = learnable_omega
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True,
                                  omega_0=first_omega_0,
                                  learnable_omega=learnable_omega))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False,
                                      omega_0=hidden_omega_0,
                                      learnable_omega=learnable_omega))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False,
                                      omega_0=hidden_omega_0,
                                      learnable_omega=learnable_omega))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

    def get_omegas(self):
        """Current omega per layer, for monitoring when learnable_omega is on."""
        omegas = []
        for layer in self.net:
            if hasattr(layer, "omega_0") and hasattr(layer, "learnable_omega"):
                if isinstance(layer.omega_0, nn.Parameter):
                    omegas.append(layer.omega_0.item())
                else:
                    omegas.append(layer.omega_0)
        return omegas

    def get_omega_L2_loss(self):
        """L2 on the learnable omegas, if any, to keep them from running away."""
        loss = 0.0
        for layer in self.net:
            if hasattr(layer, "omega_0") and hasattr(layer, "learnable_omega"):
                if layer.learnable_omega and isinstance(layer.omega_0, nn.Parameter):
                    loss = loss + layer.omega_0 ** 2
        return loss


class SirenField(nn.Module):
    """x in [0, 1]^D -> (B, n_output), with the first layer's units addressable.

    Same surface as ngp/model.py's NGPField, so the pages are twins and not
    rewrites: `n_parameters()`, `jacobian()`, `laplacian()`, and a gain vector that
    switches parts of the representation off.

    Args:
        n_input_dims / n_output_dims: D and the number of channels (3 = RGB).
        width / hidden_layers: the MLP.  hidden_layers counts the sine layers AFTER
            the first, matching `Siren` above.
        omega_0: the first layer's frequency factor, and the hidden layers' unless
            `hidden_omega_0` is given.  This is the knob for a SIREN: it sets the
            scale of the plane waves the first layer starts from.
        outermost_linear: a plain Linear on the output instead of one more sine.
        learnable_omega: let omega be trained (with `get_omega_L2_loss` to hold it).
        output_activation: "sigmoid" bounds to [0, 1] for an image, "none" leaves it.
    """

    def __init__(
        self,
        n_input_dims: int = 2,
        n_output_dims: int = 3,
        width: int = 256,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        hidden_omega_0: float | None = None,
        outermost_linear: bool = True,
        learnable_omega: bool = False,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.net = Siren(in_features=n_input_dims, hidden_features=width,
                         hidden_layers=hidden_layers, out_features=n_output_dims,
                         outermost_linear=outermost_linear,
                         first_omega_0=omega_0,
                         hidden_omega_0=omega_0 if hidden_omega_0 is None
                         else hidden_omega_0,
                         learnable_omega=learnable_omega)
        self.n_input_dims = n_input_dims
        self.width = width
        self.omega_0 = omega_0
        if output_activation == "sigmoid":
            self.out = nn.Sigmoid()
        elif output_activation == "none":
            self.out = nn.Identity()
        else:
            raise ValueError(f"unknown output_activation {output_activation!r}")

        # One gain per first-layer unit, all on.  The twin of
        # MultiResHashGrid.level_gain, and used the same way.
        self.register_buffer("unit_gain", torch.ones(width), persistent=False)

    # --------------------------------------------------------------- the ladder

    @property
    def first(self) -> SineLayer:
        return self.net.net[0]

    def omega(self) -> float:
        w = self.first.omega_0
        return float(w.item() if isinstance(w, torch.Tensor) else w)

    @torch.no_grad()
    def frequencies(self) -> torch.Tensor:
        """(width,) cycles across the unit domain, one per first-layer unit.

        The unit computes sin(w0 * (w_i . x + b_i)); along the direction of w_i that
        is a wave of w0 * |w_i| radians per unit length, so w0 * |w_i| / 2pi cycles.
        """
        return self.omega() * self.first.linear.weight.norm(dim=1) / (2 * np.pi)

    @torch.no_grad()
    def band_of(self, n_bands: int = 16) -> torch.Tensor:
        """(width,) which band each unit falls in, 0 = lowest frequency.

        Equal-count bands, not equal-width: |w_i| is concentrated by the init, so
        equal-width bins would leave most of them empty and put every unit in one.
        """
        f = self.frequencies()
        order = torch.argsort(f)
        band = torch.empty_like(order)
        edges = torch.linspace(0, self.width, n_bands + 1).round().long()
        for b in range(n_bands):
            band[order[edges[b]:edges[b + 1]]] = b
        return band

    @torch.no_grad()
    def set_band_window(self, alpha: float, n_bands: int = 16) -> None:
        """Enable bands up to `alpha`, with the fractional band faded in.

        The twin of MultiResHashGrid.set_level_window: coarse first, fine later, so
        an intensity loss is not asked to find a 40 px displacement with the finest
        waves already free to fit noise.
        """
        band = self.band_of(n_bands).float()
        self.unit_gain.copy_((alpha - band).clamp(0.0, 1.0))

    @torch.no_grad()
    def set_unit_gain(self, g: torch.Tensor) -> None:
        self.unit_gain.copy_(g.to(self.unit_gain))

    # --------------------------------------------------------------- evaluation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        if not bool(torch.all(self.unit_gain == 1)):
            h = h * self.unit_gain
        for layer in self.net.net[1:]:
            h = layer(h)
        return self.out(h)

    def n_parameters(self) -> tuple[int, int]:
        """(first-layer params, everything after it) -- the twin of NGPField's
        (encoding, decoder) split, so the two pages report the same two numbers."""
        first = sum(p.numel() for p in self.first.parameters())
        rest = sum(p.numel() for p in self.parameters()) - first
        return first, rest

    def extra_repr(self) -> str:
        f = self.frequencies()
        return (f"D={self.n_input_dims}, width={self.width}, "
                f"hidden={len(self.net.net) - 2}, omega_0={self.omega():g}, "
                f"cycles across the domain {f.min():.2f}..{f.max():.2f}")


def jacobian(model, x: torch.Tensor) -> torch.Tensor:
    """(B, out, in) df/dx by autograd, differentiable again."""
    x = x.requires_grad_(True)
    y = model(x)
    rows = [torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
            for i in range(y.shape[1])]
    return torch.stack(rows, dim=1)


def laplacian(model, x: torch.Tensor) -> torch.Tensor:
    """(B, out) sum_d d2f/dx_d^2.  Real here: a sine MLP is smooth, where a linearly
    interpolated grid has an identically zero second derivative."""
    x = x.requires_grad_(True)
    y = model(x)
    out = []
    for i in range(y.shape[1]):
        g = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, d].sum(), x, create_graph=True)[0][:, d]
                  for d in range(x.shape[1]))
        out.append(lap)
    return torch.stack(out, dim=1)


def save_checkpoint(path: str, model: SirenField, model_kwargs: dict, **extra) -> None:
    torch.save({"model_kwargs": model_kwargs, "state_dict": model.state_dict(),
                **extra}, path)


def load_checkpoint(path: str, device="cpu") -> tuple[SirenField, dict]:
    """-> (model in eval mode, the full checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SirenField(**ck["model_kwargs"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck
