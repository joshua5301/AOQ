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

THREE V PROFILES
----------------
Everything is written in u = the SIGNED distance to the nearest grid point, in
units of step: with xq = clamp(w/step, Qn, Qp) and r = xq - floor(xq),

    u = r        if r <= 0.5        (nearest level is the floor)
    u = r - 1    otherwise          (nearest level is the ceil)

so u lies in [-0.5, 0.5], the two rounding boundaries sit at u = +-0.5, and
du/dw = 1/step everywhere -- no sign() anywhere.

THREE V PROFILES, none with a width knob, so lam is QVR's only knob. Two are
genuine level-offset variances; the third is prior work run on the same code:

    measure="sr"     V     = |u| (1 - |u|)          stochastic rounding
                     dV/du = sign(u) (1 - 2|u|)

    measure="cos2"   V     = sin^2(pi u) / 4        leading Fourier mode
                     dV/du = (pi/4) sin(2 pi u)

    measure="nagel"  V     = u^2                    Nagel et al. 2022 dampening
                     dV/du = 2 u                    (NOT a variance)

They differ in what the grid point is: a smooth well (nagel), a cusp (sr), or
flat (cos2). See RELATION TO PRIOR WORK below -- nagel and sr are near mirror
images, not variations of each other.

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
factor and the relative gain weighting between coordinates is preserved. All
three profiles pull each weight toward its NEAREST grid point and all three
have V = 0 there and V = 1/4 at the boundary -- an identical barrier height,
which is what keeps lam on a comparable footing.

Where the force LIVES is the whole contrast:

    u        |dV/du| nagel    |dV/du| sr    |dV/du| cos2
    0.00     0.000            1.000         0.000          <- grid point
    0.10     0.200            0.800         0.462
    0.25     0.500            0.500         0.785          <- cos2 peaks
    0.40     0.800            0.200         0.462
    0.50     1.000            0.000         0.000          <- boundary

  nagel  convex well. Zero force at the grid point, maximal at the boundary:
         pushes hardest on the weights furthest from settling.
  sr     concave, and the mirror of nagel. Maximal force at the grid point (a
         cusp), zero at the boundary. L1-like, so a settled weight keeps
         feeling a constant-magnitude pull that flips sign as it crosses and
         jitters with amplitude set by lr.
  cos2   zero at BOTH ends, peaking mid-bin. C^infinity. Settled weights are
         left alone and the force acts on weights genuinely between levels.

Peak |dV/du| is 1.0, 1.0 and pi/4 = 0.785, so equal lam gives comparable but
not identical strength. Match force_ratio, not lam, when comparing profiles.

WHY ONE PASS. The continuous analogue is a gradient-norm penalty whose
gradient is a Hessian-vector product -- a second backward, so no cheaper than
SAM. A Gaussian measure is translation invariant, so the variance depends on w
only through grad L and detaching it leaves nothing differentiable. A
quantization grid breaks that invariance and materialises half the variance as
geometry, p(1-p), a direct function of w. Differentiating that channel alone
is free.

RELATION TO PRIOR WORK. The oscillation-dampening penalty of Nagel et al.
2022 is L_dampen = ||w_hat - clip(w)||^2, i.e. V = u^2, available here as
measure="nagel". It is NOT what QVR-sr reduces to when the gain weight is
removed -- the two are near mirror images in where the force lives:

    u          |dV/du| nagel    |dV/du| sr
    0.00       0.000            1.000        <- grid point
    0.25       0.500            0.500        <- they cross here
    0.50       1.000            0.000        <- boundary

nagel is a convex well: zero force at the grid point, maximal at the boundary,
so it pushes hardest on weights that are furthest from settling. sr is
concave: maximal force at the grid point, zero at the boundary. So
gain="const" with measure="sr" is "QVR-sr minus the sensitivity weight", an
ablation isolating what gain^2 contributes -- NOT a reproduction of prior
work, and it must not be reported as one.

One difference survives even measure="nagel". Nagel's penalty is separable,
sum_i u_i^2; QVR is sqrt(sum_i V_i gain_i^2), and the global square root
couples every coordinate through a shared denominator. gain="const" removes
the sensitivity weighting but not that coupling.

TWO ABLATION AXES
-----------------
`gain` replaces the sensitivity weight on V(u):

    "sq"     (g*step)^2   the derivation; R is the loss variance
    "abs"    |g*step|      square-root sensitivity, less peaked
    "const"  1             no sensitivity at all. NOT prior work -- see
                           RELATION TO PRIOR WORK; use measure="nagel" for that

Only "sq" makes sqrt(R) a standard deviation; the others keep the identical
machinery and change the weighting alone, which is what makes them an ablation
rather than a different method.

`apply` chooses where the term lands, and under Adam the two are genuinely
different methods, not an implementation detail:

    "decoupled"  w -= lr * lam * d sqrt(R)/dw, outside the optimizer
    "coupled"    weight.grad += lam * d sqrt(R)/dw, before optimizer.step()

Decoupled keeps its own scale, so lam is monotone under any optimizer -- the
fix AdamW makes for L2 -- but Adam normalises the TASK update to ~lr per
coordinate while leaving the penalty proportional to the raw gradient. As
training converges and ||g|| falls, the penalty's displacement falls with it
and the task's does not, so the intervention fades exactly when crystallisation
matters most. Measured on a real run at epoch 91 with lam=1, cos2: force_ratio
3.7e-3 but pull_per_step 3.6e-9 bins, i.e. ~1e-4 of a bin over the rest of
training -- inert.

Coupled fixes that: the optimizer normalises the SUM, so the penalty holds a
fixed share of the update set by force_ratio, whatever the gradient scale. The
price is saturation -- once the penalty DOMINATES, lam cancels against the lam
inside sqrt(v) and the update stops depending on lam. That only bites well
outside the intended regime (force_ratio ~ 0.05), where coupled is the better
behaved of the two.

Read force_ratio under "coupled" and pull_per_step under "decoupled";
pull_per_step is reported as NaN under "coupled" because the displacement is
then whatever the optimizer makes of the summed gradient.

STAGED IN TWO PHASES either way:

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

MEASURES = ("nagel", "sr", "cos2")
GAINS = ("sq", "abs", "const")
APPLIES = ("decoupled", "coupled")


def _gain_weight(g, step, gain):
    """The sensitivity weight multiplying V(u), per coordinate.

    "sq" is the derived one -- R is then the loss variance and lam*sqrt(R) its
    standard deviation. The other two are ablations on that weighting alone,
    everything downstream unchanged:

        sq     (g*step)^2    the derivation
        abs    |g*step|      square-root sensitivity, less peaked
        const  1             no sensitivity at all -- with measure="sr" this
                             is exactly the oscillation-dampening penalty of
                             Nagel et al. 2022, so it is the prior-work arm
    """
    if gain == "sq":
        return (g * step).pow(2)
    if gain == "abs":
        return (g * step).abs()
    return torch.ones_like(g)


def _level_variance(u, measure):
    """The V profile and its derivative, as functions of u.

    u is the signed distance to the nearest grid point in units of step, so
    u in [-0.5, 0.5] and the rounding boundaries are at +-0.5. Returns
    (V, dV/du). None of the three has a width parameter.

    "sr" and "cos2" really are level-offset variances. "nagel" is NOT -- it is
    the prior-work dampening penalty put through the same machinery, kept here
    so the comparison runs on identical code.

    Kept a free function so the self-check exercises exactly this code rather
    than a re-derivation of it.
    """
    if measure == "nagel":
        # L_dampen = ||w_hat - clip(w)||^2 of Nagel et al. 2022, in bin units.
        return u * u, 2.0 * u

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
    def __init__(self, model, lam=0.0, measure="sr", gain="sq",
                 apply_mode="decoupled", eps=1e-12):
        if lam < 0.0:
            raise ValueError("lam must be non-negative, got {}".format(lam))
        if measure not in MEASURES:
            raise ValueError("measure must be one of {}, got {!r}".format(MEASURES, measure))
        if gain not in GAINS:
            raise ValueError("gain must be one of {}, got {!r}".format(GAINS, gain))
        if apply_mode not in APPLIES:
            raise ValueError("apply must be one of {}, got {!r}".format(APPLIES, apply_mode))

        self.lam = float(lam)
        self.measure = measure
        self.gain = gain
        self.apply_mode = apply_mode
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
            wgt = _gain_weight(g, step, self.gain)
            R += float((var * wgt)[valid].double().sum().item())
            g_sq += float(g.double().pow(2).sum().item())
            pending.append((module, torch.where(valid, wgt * dvar, torch.zeros_like(g)), step))

        std = math.sqrt(max(R, 0.0))
        # Floor matters: as the network crystallises R -> 0 and a raw 1/sqrt(R)
        # would blow up on the last few stragglers.
        scale = self.lam / (2.0 * max(std, self.eps))

        force_sq, pull, n = 0.0, 0.0, 0
        for module, dR, step in pending:
            delta = dR * scale
            force_sq += float(delta.double().pow(2).sum().item())
            # step broadcasts, so this stays right for a per-channel step_size.
            pull += float((delta.double().abs() / step.double()).sum().item())
            n += delta.numel()
            if self.apply_mode == "coupled":
                # Ride the optimizer: Adam then normalises the SUM, so the
                # penalty keeps a fixed share of the update as the task
                # gradient decays. apply() has nothing left to do.
                module.weight.grad.add_(delta.to(module.weight.grad.dtype))
            else:
                self._staged.append((module, delta))

        self.stats = {
            "lam": self.lam,
            "std": std,
            # Share of the raw gradient the penalty contributes. This IS the
            # operative knob under apply="coupled", where the optimizer
            # normalises the sum and only the ratio survives.
            "force_ratio": (force_sq ** 0.5) / (g_sq ** 0.5) if g_sq > 0 else float("nan"),
            # Fraction of a quantization bin the penalty drags a weight per
            # step. Only defined for apply="decoupled" -- under "coupled" the
            # displacement is whatever the optimizer makes of the summed
            # gradient, so reporting lr*|delta| there would be a fiction.
            "pull_per_step": float("nan"),
            "_pull_unit_lr": pull / n if n else float("nan"),
        }

    @torch.no_grad()
    def apply(self, lr):
        """Write the decoupled update. Call right after optimizer.step().

        A no-op under apply="coupled", where stage() already added the term to
        weight.grad and the optimizer has consumed it.
        """
        for module, delta in self._staged:
            module.weight.data.add_(delta, alpha=-lr)
        self._staged = []
        if self.stats:
            unit = self.stats.pop("_pull_unit_lr")
            if self.apply_mode == "decoupled":
                self.stats["pull_per_step"] = unit * lr

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

    # 4. Where the force lives -- the whole point of having three profiles.
    #    sr has to be probed in the limit at u=0 because torch.sign(0) == 0
    #    hands back the 0 subgradient of the cusp, admissible but not the force.
    WANT = {  # (grid point, boundary)
        "nagel": ("zero", "max"),
        "sr": ("max", "zero"),
        "cos2": ("zero", "zero"),
    }
    for measure in MEASURES:
        peak = _level_variance(grid, measure)[1].abs().max().item()
        at0 = _level_variance(torch.zeros(1, dtype=torch.float64), measure)[1].abs().item()
        near0 = _level_variance(
            torch.tensor([-1e-9, 1e-9], dtype=torch.float64), measure)[1].abs().max().item()
        edge = _level_variance(
            torch.tensor([-0.5, 0.5], dtype=torch.float64), measure)[1].abs().max().item()
        want_grid, want_edge = WANT[measure]
        if want_grid == "zero":
            assert at0 <= 1e-12 * peak, (measure, "grid point force-free", at0 / peak)
        else:
            assert near0 >= (1.0 - 1e-6) * peak, (measure, "grid point maximal", near0 / peak)
        if want_edge == "zero":
            assert edge <= 1e-12 * peak, (measure, "boundary force-free", edge / peak)
        else:
            assert edge >= (1.0 - 1e-6) * peak, (measure, "boundary maximal", edge / peak)
        print("  {:<5s} peak|dV/du|={:.4f}  |force|/peak: grid point={:.3f} ({})"
              "  boundary={:.3f} ({})".format(
                  measure, peak, near0 / peak, want_grid, edge / peak, want_edge))

    # nagel and sr are near mirror images: they cross at |u| = 1/4.
    q = torch.tensor([0.25], dtype=torch.float64)
    dn = abs(float(_level_variance(q, "nagel")[1]))
    ds = abs(float(_level_variance(q, "sr")[1]))
    assert abs(dn - 0.5) < 1e-12 and abs(ds - 0.5) < 1e-12, (dn, ds)
    print("  nagel vs sr cross at |u|=0.25: both |dV/du| = {:.3f}".format(dn))

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

    # 7. gain ablations: the weight multiplying V, everything else identical.
    #    "const" + "sr" reproduces the Nagel dampening penalty exactly.
    g = torch.randn(2048, dtype=torch.float64)
    step = torch.tensor(0.3, dtype=torch.float64)
    w = {name: _gain_weight(g, step, name) for name in GAINS}
    assert torch.allclose(w["sq"], (g * step) ** 2)
    assert torch.allclose(w["abs"], (g * step).abs())
    assert torch.equal(w["const"], torch.ones_like(g))
    assert torch.allclose(w["sq"], w["abs"] ** 2), "abs must be the sqrt of sq"
    print("  gain weights: sq=|g*step|^2, abs=|g*step|, const=1  (abs^2 == sq)")

    print("qvr self-check OK")


if __name__ == "__main__":
    _self_check()
