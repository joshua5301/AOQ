"""QVR - Quantization-aware Variance Regularization.

Penalises the spread of the loss induced by rounding. To first order, with
gain_i = g_i * step_i the loss change from flipping coordinate i and p_i its
flip probability under the chosen measure,

    R(w)    = sum_i p_i (1 - p_i) gain_i^2          the variance
    penalty = lam * sqrt(R)                          the STANDARD DEVIATION

sqrt(R), not R, is what the likelihood-ball expansion produces:

    max over ball(B)  ~=  E[L] + sqrt(2B) * Std(L),      lam = sqrt(2B)

so lam is the ball radius, and this is the exact grid analogue of SAM's
gradient-norm penalty (rho*||g|| = sqrt(2B) * sigma * ||g||). The name says
variance because that is the quantity being decomposed; the penalty is its
square root.

TWO MEASURES
------------
Everything is written in u = the SIGNED distance to the nearest grid point, in
units of step: with xq = clamp(w/step, Qn, Qp) and r = xq - floor(xq),

    u = r        if r <= 0.5        (nearest level is the floor)
    u = r - 1    otherwise          (nearest level is the ceil)

so u lies in [-0.5, 0.5], the two rounding boundaries sit at u = +-0.5, and
du/dw = 1/step everywhere -- no sign() anywhere.

Let V be the variance of the LEVEL OFFSET d in {0, +1, -1}. Then V is the
p(1-p) of the two-outcome case and its generalisation otherwise:

    measure="sr"        p    = |u|                        stochastic rounding
                        V    = p (1 - p)
                        dV/du = sign(u) (1 - 2|u|)

    measure="gaussian"  p_+  = Phi((u - 0.5) / alpha)     jitter crosses +1/2
                        p_-  = Phi((-u - 0.5) / alpha)    jitter crosses -1/2
                        s    = p_+ - p_-  = E[d]
                        V    = p_+ + p_- - s^2            = E[d^2] - E[d]^2
                        dV/du = [phi_+ (1 - 2s) - phi_- (1 + 2s)] / alpha

with alpha = sigma / step and phi_+- = phi((0.5 -+ u) / alpha). The level
offset is truncated at +-1, which costs P(|jitter| > 1.5 step) -- 4e-4 at
sigma = 0.3, less below. Keep sigma <~ 0.2 if that matters.

The Gaussian arm counts BOTH boundaries. A one-sided p = Phi(-m/sigma) on the
nearest boundary alone is wrong at a grid point, where the two boundaries are
symmetric and the force must vanish exactly: measured against the peak force,
the one-sided residual there is 0.0% at sigma=0.10 but 53% at sigma=0.30. It
reduces to the one-sided form when p_- ~ 0, which is why small sigma hid it.

SR carries no width knob -- the measure fixes it. Gaussian takes an explicit
sigma, which is the jitter width and therefore the reach of the penalty:
coordinates past ~3 sigma from a boundary feel essentially nothing, where SR
keeps pulling across the whole bin. sigma is in UNITS OF step_size, so it
stays meaningful as LSQ learns a different step.

GRADIENT (gain detached, so only the position channel flows)
------------------------------------------------------------
    dR/dw        = gain^2 * (dV/du) / step
    d sqrt(R)/dw = dR/dw / (2 sqrt(R))

sqrt(R) is one global scalar, so it rescales every coordinate by the same
factor and the relative gain^2 weighting between coordinates is preserved.
Both forms pull each weight toward its NEAREST grid point. u = 0 is a stable
equilibrium and EXACTLY force-free under both measures -- that symmetry is the
test a one-sided Gaussian fails. The boundary u = +-0.5 is force-free exactly
under sr, but only as sigma -> 0 under gaussian, since the far boundary then
sits a whole step away rather than symmetrically: |force|/peak there is 9e-22,
1.6e-5 and 1.9e-2 at sigma = 0.1, 0.2 and 0.3.

WHY ONE PASS. The continuous analogue is a gradient-norm penalty whose
gradient is a Hessian-vector product -- a second backward, so no cheaper than
SAM. A Gaussian measure is translation invariant, so the variance depends on w
only through grad L and detaching it leaves nothing differentiable. A
quantization grid breaks that invariance and materialises half the variance as
geometry, p(1-p), a direct function of w. Differentiating that channel alone
is free.

RELATION TO PRIOR WORK. Drop the gain^2 weight from the SR form and
sum_i r_i(1-r_i) is the oscillation-dampening penalty of Nagel et al. 2022.
That heuristic is the special case with uniform sensitivity; what QVR adds is
a sensitivity weight, derived rather than assumed.

DECOUPLED, AND IN TWO PHASES. The update is applied as
w -= lr * lam * d sqrt(R)/dw, outside the optimizer rather than added to
weight.grad: Adam normalises by its own second-moment estimate, so a coupled
penalty saturates -- once it dominates the gradient, lam cancels against the
lam inside sqrt(v) and the update stops depending on lam at all. Same fix
AdamW makes for L2.

That forces the split into stage() and apply():

    loss.backward()
    qvr.stage()          # gain AND geometry, both read at w_t
    optimizer.step()     # w_t -> w_{t+1}
    qvr.apply(lr)        # w -= lr * delta

Doing it all after optimizer.step() would read the gradient at w_t but the
position at w_{t+1}, so the two channels of grad sqrt(R) would be evaluated
one step apart. The error grows with the size of the task update.

LIMITS. gain is detached, so there is no desensitisation channel (that would
need an HVP). gain^2 is a minibatch estimate, so the penalty charges gradient
noise alongside true sensitivity, and sqrt(R) in the denominator is itself a
batch estimate, biased slightly down by Jensen.

Under SR, V = |u|(1-|u|) has a kink at u = 0: the force does not switch off at
a grid point, it reverses, so a settled weight jitters around the grid point
with amplitude ~pull_per_step. That is the measure being honest (SR really is
indifferent there), not a bug, but it is why the Gaussian arm exists. lam is a
plain attribute, so a caller that wants a ramp can just assign to qvr.lam.

Self-check:  python3 AO_QAT/quan/qvr.py
"""

import math

import torch

from .func import QuanConv2d, QuanLinear
from .quantizer import LSQ

MEASURES = ("sr", "gaussian")
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z):
    return _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def _Phi(z):
    return 0.5 * torch.erfc(-z / math.sqrt(2.0))


def _level_variance(u, measure, sigma):
    """Variance of the level offset, and its derivative, as functions of u.

    u is the signed distance to the nearest grid point in units of step, so
    u in [-0.5, 0.5] and the rounding boundaries are at +-0.5. sigma is also
    in units of step. Returns (V, dV/du).

    Kept a free function so the self-check exercises exactly this code rather
    than a re-derivation of it.
    """
    if measure == "sr":
        p = u.abs()
        return p * (1.0 - p), torch.sign(u) * (1.0 - 2.0 * p)

    # Both boundaries: a one-sided form is wrong at a grid point, where the
    # two are symmetric and the force must cancel exactly.
    p_up = _Phi((u - 0.5) / sigma)
    p_dn = _Phi((-u - 0.5) / sigma)
    s = p_up - p_dn  # E[level offset]
    var = p_up + p_dn - s * s  # E[d^2] - E[d]^2
    dvar = (
        _phi((0.5 - u) / sigma) * (1.0 - 2.0 * s)
        - _phi((0.5 + u) / sigma) * (1.0 + 2.0 * s)
    ) / sigma
    return var, dvar


class QVR:
    def __init__(self, model, lam=0.0, measure="sr", sigma=0.1, eps=1e-12):
        if lam < 0.0:
            raise ValueError("lam must be non-negative, got {}".format(lam))
        if measure not in MEASURES:
            raise ValueError("measure must be one of {}, got {!r}".format(MEASURES, measure))
        if measure == "gaussian" and sigma <= 0.0:
            raise ValueError("gaussian needs sigma > 0, got {}".format(sigma))

        self.lam = float(lam)
        self.measure = measure
        self.sigma = float(sigma)  # in units of step_size
        self.eps = float(eps)
        self.stats = {}
        self._staged = []
        self.layers = [
            m
            for m in model.modules()
            if isinstance(m, (QuanConv2d, QuanLinear))
            and isinstance(getattr(m, "quan_w_fn", None), LSQ)
        ]
        if not self.layers:
            raise RuntimeError("QVR found no LSQ-quantized layers")

    def _geometry(self, module):
        """Level-offset variance V and dV/dw, per coordinate."""
        q = module.quan_w_fn
        step = q.step_size.detach().float()
        x = module.weight.detach().float() / step
        xq = x.clamp(q.thd_neg, q.thd_pos)
        r = xq - xq.floor()
        # Signed distance to the nearest grid point, so du/dw = 1/step
        # everywhere and the two boundaries sit symmetrically at +-0.5.
        u = r - (r > 0.5).to(r.dtype)
        valid = (x >= q.thd_neg) & (x <= q.thd_pos)

        var, dvar_du = _level_variance(u, self.measure, self.sigma)
        return step, var, dvar_du / step, valid

    @torch.no_grad()
    def stage(self):
        """Compute the update from w_t. Call after backward(), before step()."""
        self._staged = []
        self.stats = {}
        if self.lam <= 0.0:
            return

        pending, R, g_sq = [], 0.0, 0.0
        for module in self.layers:
            grad = module.weight.grad
            if grad is None:
                continue
            g = grad.detach().float()
            if not torch.isfinite(g).all():
                continue
            step, var, dvar, valid = self._geometry(module)
            gain2 = (g * step).pow(2)
            R += float((var * gain2)[valid].double().sum().item())
            g_sq += float(g.double().pow(2).sum().item())
            pending.append((module, torch.where(valid, gain2 * dvar, torch.zeros_like(g)), step))

        std = math.sqrt(max(R, 0.0))
        # Floor matters: as the network crystallises R -> 0 and a raw 1/sqrt(R)
        # would blow up on the last few stragglers.
        scale = self.lam / (2.0 * max(std, self.eps))

        force_sq, pull, n = 0.0, 0.0, 0
        for module, dR, step in pending:
            delta = dR * scale
            self._staged.append((module, delta))
            force_sq += float(delta.double().pow(2).sum().item())
            # step broadcasts, so this stays right for a per-channel step_size.
            pull += float((delta.double().abs() / step.double()).sum().item())
            n += delta.numel()

        self.stats = {
            "lam": self.lam,
            "std": std,
            # Penalty gradient next to the task gradient. Scale-free reference.
            "force_ratio": (force_sq ** 0.5) / (g_sq ** 0.5) if g_sq > 0 else float("nan"),
            # Fraction of a quantization bin the penalty drags a weight per
            # step. This is the number to tune lam by. Filled in by apply().
            "pull_per_step": float("nan"),
            "_pull_unit_lr": pull / n if n else float("nan"),
        }

    @torch.no_grad()
    def apply(self, lr):
        """Apply the staged update. Call right after optimizer.step()."""
        for module, delta in self._staged:
            module.weight.data.add_(delta, alpha=-lr)
        self._staged = []
        if self.stats:
            self.stats["pull_per_step"] = self.stats.pop("_pull_unit_lr") * lr

    @torch.no_grad()
    def step(self, lr):
        """Apply the penalty. Call right after optimizer.step().

        Two loops: sqrt(R) is one scalar for the whole network, so R has to be
        summed over every layer before any coordinate can be scaled.
        """
        self.stats = {}
        if self.lam <= 0.0:
            return

        pending, R, g_sq = [], 0.0, 0.0
        for module in self.layers:
            grad = module.weight.grad
            if grad is None:
                continue
            g = grad.detach().float()
            if not torch.isfinite(g).all():
                continue
            step, var, dvar, valid = self._geometry(module)
            gain2 = (g * step).pow(2)
            R += float((var * gain2)[valid].double().sum().item())
            g_sq += float(g.double().pow(2).sum().item())
            pending.append((module, torch.where(valid, gain2 * dvar, torch.zeros_like(g)), step))

        std = math.sqrt(max(R, 0.0))
        # Floor matters: as the network crystallises R -> 0 and a raw 1/sqrt(R)
        # would blow up on the last few stragglers.
        scale = self.lam / (2.0 * max(std, self.eps))

        pull, n, force_sq = 0.0, 0, 0.0
        for module, dR, step in pending:
            delta = dR * scale
            module.weight.data.add_(delta, alpha=-lr)
            force_sq += float(delta.double().pow(2).sum().item())
            s = float(step.reshape(-1)[0].item())
            if s:
                pull += float(delta.double().abs().sum().item()) * lr / s
                n += delta.numel()

        self.stats = {
            "lam": self.lam,
            "std": std,
            # Penalty gradient next to the task gradient. Scale-free reference.
            "force_ratio": (force_sq ** 0.5) / (g_sq ** 0.5) if g_sq > 0 else float("nan"),
            # Fraction of a quantization bin the penalty drags a weight per
            # step. This is the number to tune lam by.
            "pull_per_step": pull / n if n else float("nan"),
        }

    def summary(self):
        if not self.stats:
            return "qvr: inactive (lam=0)"
        return (
            "qvr[{}] lam={lam:.4e} std={std:.4e} force/grad={force_ratio:.4e} "
            "pull/step={pull_per_step:.3e}".format(self.measure, **self.stats)
        )


def _self_check():
    """Two independent claims: the derivative is right, and the measure is."""
    torch.manual_seed(0)
    qn, qp = -2, 1

    # 1. dV/du against autograd, through the SAME _level_variance code.
    for measure in MEASURES:
        for sigma in (0.1, 0.3):
            u = (torch.rand(4096, dtype=torch.float64) - 0.5).requires_grad_(True)
            var, dvar = _level_variance(u, measure, sigma)
            (auto,) = torch.autograd.grad(var.sum(), u)
            err = (auto - dvar.detach()).abs().max().item()
            assert err <= 1e-9 * max(dvar.abs().max().item(), 1.0), (measure, sigma, err)
            print("  {:<8s} sigma={:<4g} max|autograd - analytic dV/du| = {:.3e}".format(
                measure, sigma, err))

    # 2. The full d sqrt(R)/dw chain, end to end through w.
    for measure in MEASURES:
        for step_val in (0.05, 1.9):
            w = torch.randn(4096, dtype=torch.float64, requires_grad=True)
            g = torch.randn(4096, dtype=torch.float64)
            step = torch.tensor(step_val, dtype=torch.float64)
            xq = (w / step).clamp(qn, qp)
            r = xq - xq.floor()
            u = r - (r > 0.5).to(r.dtype)
            var, _ = _level_variance(u, measure, 0.15)
            R = (var * (g * step).pow(2)).sum()
            (auto,) = torch.autograd.grad(R.sqrt(), w)

            _, dvar = _level_variance(u.detach(), measure, 0.15)
            in_range = ((w / step >= qn) & (w / step <= qp)).double()
            analytic = (g * step).pow(2) * (dvar / step) * in_range / (2.0 * R.detach().sqrt())
            err = (auto - analytic).abs().max().item()
            assert err <= 1e-8 * max(analytic.abs().max().item(), 1.0), (measure, step_val, err)
            print("  {:<8s} step={:<5g} max|autograd - analytic dsqrt(R)/dw| = {:.3e}".format(
                measure, step_val, err))

    # 3. Boundary is an unstable equilibrium; GRID POINTS are stable ones.
    #    The grid-point check is what separates a two-sided measure from a
    #    one-sided one -- the boundary check passes either way.
    for measure in MEASURES:
        edge = []
        for sigma in (0.1, 0.2, 0.3):
            u = torch.tensor([-0.5, -1e-9, 0.0, 1e-9, 0.5], dtype=torch.float64)
            _, dvar = _level_variance(u, measure, sigma)
            force = -dvar  # descent direction, gain^2 > 0
            grid = torch.linspace(-0.5, 0.5, 2001, dtype=torch.float64)
            peak = _level_variance(grid, measure, sigma)[1].abs().max().item()
            assert force[1] > 0 and force[3] < 0, (measure, "grid point must restore")
            # EXACT, and this is the discriminating test: at u=0 the two
            # boundaries are symmetric, so p_+ == p_-, s == 0 and the two phi
            # terms cancel identically. A one-sided Phi(-m/sigma) fails it by
            # 53% of peak at sigma=0.3.
            assert abs(force[2]) < 1e-12 * peak, (
                measure, sigma, "grid point must be an exact equilibrium",
                abs(force[2].item()) / peak)
            edge.append(abs(force[0].item()) / peak)
            print("  {:<8s} sigma={:<4g} peak={:.3e}  |force|/peak at grid point={:.1e}"
                  "  at boundary={:.1e}".format(
                      measure, sigma, peak, abs(force[2].item()) / peak, edge[-1]))
        # The boundary is an EXACT equilibrium only for sr. Under gaussian the
        # far boundary sits one whole step away rather than symmetrically, so
        # it contributes, and only vanishes as sigma -> 0. Assert that decay
        # rather than an absolute tolerance -- at sigma=0.3 the far boundary is
        # a mere 3.3 sigma off and is worth ~2% of peak.
        if measure == "sr":
            assert max(edge) < 1e-12, edge
        else:
            assert edge[0] < edge[1] < edge[2], ("boundary residual must fall with sigma", edge)
            assert edge[0] < 1e-9, edge

    # 4. Locality: gaussian dies deep inside the bin, sr does not.
    u = torch.tensor([0.05], dtype=torch.float64)  # 0.45 from either boundary
    sr = abs(float(_level_variance(u, "sr", 0.0)[1]))
    gau = abs(float(_level_variance(u, "gaussian", 0.05)[1]))
    assert gau < 1e-6 * sr, (gau, sr)
    print("  locality at u=0.05 (9 sigma from a boundary): sr={:.3e}  gaussian={:.3e}".format(
        sr, gau))

    print("qvr self-check OK")


if __name__ == "__main__":
    _self_check()
