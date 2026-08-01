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

Let V be the variance of the LEVEL OFFSET. NEITHER ARM HAS A WIDTH KNOB, so
lam is QVR's only hyperparameter:

    measure="sr"     V     = |u| (1 - |u|)          stochastic rounding
                     dV/du = sign(u) (1 - 2|u|)

    measure="cos2"   V     = sin^2(pi u) / 4        leading Fourier mode
                     dV/du = (pi/4) sin(2 pi u)

WHY sin^2 AND NOT A GAUSSIAN. V is periodic in u, so it has a Fourier series.
For a Gaussian latent jitter of width sigma the pushforward variance is a
theta series with coefficients decaying like exp(-2 pi^2 sigma^2 k^2); the
second mode relative to the first is 3.4e-1, 1.6e-1, 6.1e-2, 2.3e-2 at
sigma = 0.15, 0.20, 0.25, 0.30. Past sigma ~ 0.2 only k=1 survives, and that
mode is 1 - cos(2 pi u) = 2 sin^2(pi u). Single-mode error at sigma = 0.25 is
3.1% of the swing. So sin^2 is not an arbitrary smooth kernel: it is the
sigma-independent LIMIT of the Gaussian arm, and fixing the arm to it removes
the knob instead of picking a value for it.

Secondary reading (not the derivation): V = sin^2(pi u)/4 is also exactly the
pushforward variance of a dither with density (pi/2) cos(pi t) on
[-1/2, 1/2], i.e. one LSB wide. That is a convenient description, not a
standard dither type, so it should not be cited as one.

Why one LSB is the ceiling: the standard hierarchy (RPDF -> TPDF -> nRPDF ->
Gaussian) widens the support, and the position signal dies at the second step.
Under TPDF -- triangular, two LSB -- the pushforward variance is EXACTLY
constant, V(u) = 1/4 for every u, so the penalty vanishes identically. With
a = 1/2 - u that falls out as E[d] = u, E[d^2] = 1/2 - a + a^2, hence
Var = 1/4. It is the classical dither theorem (RPDF makes the first error
moment signal-independent, TPDF the second) working against us: that
dependence is the entire signal here. Both surviving arms sit exactly at the
one-LSB ceiling and differ only in the shape of the density.

GRADIENT (gain detached, so only the position channel flows)
------------------------------------------------------------
    dR/dw        = gain^2 * (dV/du) / step
    d sqrt(R)/dw = dR/dw / (2 sqrt(R))

sqrt(R) is one global scalar, so it rescales every coordinate by the same
factor and the relative gain^2 weighting between coordinates is preserved.
Both forms pull each weight toward its NEAREST grid point, both have V = 0
there and V = 1/4 at the boundary -- an identical barrier height, which is
what makes lam comparable across the arms -- and the boundary is an exact
equilibrium in both.

They differ in WHERE the force lives, and that contrast is why both exist:

    u        V(sr)    dV/du(sr)     V(cos2)   dV/du(cos2)
    0.00     0.0000   +1.0000       0.0000    +0.0000
    0.10     0.0900   +0.8000       0.0239    +0.4616
    0.25     0.1875   +0.5000       0.1250    +0.7854
    0.40     0.2400   +0.2000       0.2261    +0.4616
    0.50     0.2500    0.0000       0.2500    +0.0000

  sr    force is MAXIMAL at the grid point (V is a V-shaped well with a kink)
        and falls linearly to zero at the boundary. L1-like: a settled weight
        keeps feeling a constant-magnitude pull that flips sign as it crosses,
        so it jitters around the grid point with amplitude set by lr.
  cos2  force is zero at the grid point AND at the boundary, peaking mid-bin
        at |u| = 0.25. C^infinity everywhere. Settled weights are left alone
        and the force acts on weights genuinely between levels.

Peak |dV/du| differs by 4/pi (1.0 against 0.785) while the barrier matches, so
equal lam gives comparable but not identical strength. Match force_ratio, not
lam, when comparing the arms.

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

STEP_SIZE IS DETACHED, ON PURPOSE. R depends on step twice -- through
gain = g * step and through r = frac(w / step) -- so dR/dstep is not zero.
It is dropped anyway, and nothing here writes to step_size.grad: the geometry
reads q.step_size.detach(), the whole update runs under no_grad, and it lands
on weight.data alone.

The reason is that dR/dstep points almost entirely at shrinking step. R scales
like step^2 through gain^2, so descent on lam*sqrt(R) would shrink the
quantization scale until each flip stopped mattering -- a trivial way to make
the penalty small that has nothing to do with rounding robustness, and it
would quietly wreck the quantizer LSQ is trying to learn. The penalty is meant
to move the LATENT WEIGHTS relative to a grid the task loss chooses, so the
grid is held fixed while it does. step_size still trains normally, from the
task gradient, at the weights QVR has moved.

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

MEASURES = ("sr", "cos2")


def _level_variance(u, measure):
    """Variance of the level offset, and its derivative, as functions of u.

    u is the signed distance to the nearest grid point in units of step, so
    u in [-0.5, 0.5] and the rounding boundaries are at +-0.5. Returns
    (V, dV/du). Neither arm has a width parameter.

    Kept a free function so the self-check exercises exactly this code rather
    than a re-derivation of it.
    """
    if measure == "sr":
        p = u.abs()
        # torch.sign(0) == 0, so u == 0 returns the 0 subgradient of the kink.
        # Any value in [-1, 1] is admissible there; 0 is the convenient one.
        return p * (1.0 - p), torch.sign(u) * (1.0 - 2.0 * p)

    # Leading Fourier mode: V = sin^2(pi u)/4, dV/du = (pi/4) sin(2 pi u).
    return (
        torch.sin(math.pi * u).pow(2) / 4.0,
        (math.pi / 4.0) * torch.sin(2.0 * math.pi * u),
    )


class QVR:
    def __init__(self, model, lam=0.0, measure="sr", eps=1e-12):
        if lam < 0.0:
            raise ValueError("lam must be non-negative, got {}".format(lam))
        if measure not in MEASURES:
            raise ValueError("measure must be one of {}, got {!r}".format(MEASURES, measure))

        self.lam = float(lam)  # the only hyperparameter, in both arms
        self.measure = measure
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

        var, dvar_du = _level_variance(u, self.measure)
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
    grid = torch.linspace(-0.5, 0.5, 20001, dtype=torch.float64)

    # 1. dV/du against autograd, through the SAME _level_variance code.
    for measure in MEASURES:
        u = (torch.rand(4096, dtype=torch.float64) - 0.5).requires_grad_(True)
        var, dvar = _level_variance(u, measure)
        (auto,) = torch.autograd.grad(var.sum(), u)
        err = (auto - dvar.detach()).abs().max().item()
        assert err <= 1e-9 * max(dvar.abs().max().item(), 1.0), (measure, err)
        print("  {:<5s} max|autograd - analytic dV/du| = {:.3e}".format(measure, err))

    # 2. The full d sqrt(R)/dw chain, end to end through w, at two steps.
    for measure in MEASURES:
        for step_val in (0.05, 1.9):
            w = torch.randn(4096, dtype=torch.float64, requires_grad=True)
            g = torch.randn(4096, dtype=torch.float64)
            step = torch.tensor(step_val, dtype=torch.float64)
            xq = (w / step).clamp(qn, qp)
            r = xq - xq.floor()
            u = r - (r > 0.5).to(r.dtype)
            var, _ = _level_variance(u, measure)
            R = (var * (g * step).pow(2)).sum()
            (auto,) = torch.autograd.grad(R.sqrt(), w)

            _, dvar = _level_variance(u.detach(), measure)
            in_range = ((w / step >= qn) & (w / step <= qp)).double()
            analytic = (g * step).pow(2) * (dvar / step) * in_range / (2.0 * R.detach().sqrt())
            err = (auto - analytic).abs().max().item()
            assert err <= 1e-8 * max(analytic.abs().max().item(), 1.0), (measure, step_val, err)
            print("  {:<5s} step={:<5g} max|autograd - analytic dsqrt(R)/dw| = {:.3e}".format(
                measure, step_val, err))

    # 3. Sign: both arms restore toward the nearest grid point.
    for measure in MEASURES:
        u = torch.tensor([-0.25, 0.25], dtype=torch.float64)
        force = -_level_variance(u, measure)[1]
        assert force[0] > 0 and force[1] < 0, (measure, "must restore to u=0")

    # 4. Where the force lives -- the whole point of having two arms.
    #    cos2 switches off AT the grid point; sr is maximal there. sr has to
    #    be probed in the limit because torch.sign(0) == 0 hands back the 0
    #    subgradient of the kink, which is admissible but not the force.
    for measure in MEASURES:
        peak = _level_variance(grid, measure)[1].abs().max().item()
        at0 = _level_variance(torch.zeros(1, dtype=torch.float64), measure)[1].abs().item()
        near0 = _level_variance(
            torch.tensor([-1e-9, 1e-9], dtype=torch.float64), measure)[1].abs().max().item()
        edge = _level_variance(
            torch.tensor([-0.5, 0.5], dtype=torch.float64), measure)[1].abs().max().item()
        if measure == "cos2":
            assert at0 <= 1e-12 * peak, (measure, "grid point must be force-free", at0 / peak)
            assert near0 <= 1e-7 * peak, (measure, "and smoothly so", near0 / peak)
        else:
            assert near0 >= (1.0 - 1e-6) * peak, (measure, "grid point maximal", near0 / peak)
        assert edge <= 1e-12 * peak, (measure, "boundary must be force-free", edge / peak)
        print("  {:<5s} peak|dV/du|={:.4f}  |force|/peak just off the grid point={:.3f}"
              "  at boundary={:.1e}".format(measure, peak, near0 / peak, edge / peak))

    # 5. Barrier height is identical, which is what makes lam comparable.
    for measure in MEASURES:
        v0 = float(_level_variance(torch.zeros(1, dtype=torch.float64), measure)[0])
        v_edge = float(_level_variance(torch.tensor([0.5], dtype=torch.float64), measure)[0])
        assert abs(v0) < 1e-12 and abs(v_edge - 0.25) < 1e-12, (measure, v0, v_edge)
        print("  {:<5s} V(0)={:.1e}  V(1/2)={:.6f}  barrier={:.6f}".format(
            measure, v0, v_edge, v_edge - v0))

    # 6. The two claims the docstring makes about WHY cos2, computed rather
    #    than asserted. The Gaussian pushforward variance is taken EXACTLY --
    #    V = E[d^2] - E[d]^2 summed over levels -- because the -E[d]^2 term
    #    mixes Fourier modes, so no single theta series reproduces it.
    def _gauss_V(u, sigma, K=12):
        ks = torch.arange(-K, K + 1, dtype=torch.float64).unsqueeze(1)
        Phi = lambda z: 0.5 * torch.erfc(-z / math.sqrt(2.0))
        p = Phi((ks + 0.5 - u) / sigma) - Phi((ks - 0.5 - u) / sigma)
        m1 = (p * ks).sum(0)
        m2 = (p * ks * ks).sum(0)
        return m2 - m1 * m1

    n = 4096
    u = torch.arange(n, dtype=torch.float64) / n - 0.5
    cos2_v = _level_variance(u, "cos2")[0]
    for sigma, want_ratio, want_err in ((0.15, 3.4e-1, 0.168), (0.25, 6.1e-2, 0.031)):
        v = _gauss_V(u, sigma)
        swing = float(_gauss_V(torch.tensor([0.5], dtype=torch.float64), sigma)
                      - _gauss_V(torch.zeros(1, dtype=torch.float64), sigma))
        c = torch.fft.rfft(v - v.mean()).abs() / n
        ratio = float(c[2] / c[1])
        approx = (cos2_v - cos2_v.mean()) * (swing / 0.25)
        err = float((v - v.mean() - approx).abs().max()) / abs(swing)
        assert abs(ratio - want_ratio) < 0.05 * want_ratio, (sigma, ratio, want_ratio)
        assert abs(err - want_err) < 0.1 * want_err, (sigma, err, want_err)
        print("  gaussian sigma={:<5g} c2/c1={:.3e}  cos2 vs exact: {:.1%} of swing".format(
            sigma, ratio, err))
    # TPDF, two LSB: V(u) == 1/4 identically, so the penalty would vanish.
    a_ = 0.5 - u
    tpdf = 0.5 - a_ + a_ * a_ - u * u
    assert (tpdf - 0.25).abs().max().item() < 1e-12
    print("  TPDF (2 LSB): max|V - 1/4| = {:.1e}  -> position signal gone".format(
        (tpdf - 0.25).abs().max().item()))

    print("qvr self-check OK")


if __name__ == "__main__":
    _self_check()
