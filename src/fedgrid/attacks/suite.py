"""Attack suite: naive baselines plus the defense-aware adaptive attacker.

Everything here works in *delta space*. ``ClientUpdate.params`` carries full
parameters by contract, but a poisoning transform applied to full parameters is
meaningless -- negating them negates the global model every client shares, which
no defense needs a fidelity score to notice. The quantity a client actually
contributes is ``params - ctx.ref``, so each attack reconstructs that delta,
poisons it, and re-adds ``ctx.ref``. This is also what makes the attacks
comparable to each other and to the fidelity score, which the pipeline computes
on the same deltas.

``ClientUpdate.attack`` is set to the attack's tag on the way out. It is ground
truth for evaluation plots only; nothing in ``fedgrid.consensus`` reads it, and a
test that let the defense see it would be measuring nothing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from fedgrid.attacks.base import Attack, AttackContext
from fedgrid.consensus.fidelity import flatten, unflatten

if TYPE_CHECKING:
    from fedgrid.fl.engine import ClientUpdate

__all__ = ["AdaptiveCosine", "GaussianNoise", "LabelFlip", "SignFlip"]

_NORM_EPS = 1e-12


class SignFlip(Attack):
    """Model poisoning: submit ``scale * (-update)`` -- loud, easily flagged baseline.

    The reference "obviously Byzantine" attack: the client trains honestly and
    then reverses its step, so the submitted delta is anti-parallel to the honest
    consensus direction and its fidelity score sits near the maximum of 2. Its
    role in the paper is to be the attack every defense catches, so that what the
    adaptive attacker gets away with is measured against something.
    """

    tag = "sign_flip"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def poison(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate:
        ref = flatten(ctx.ref)
        delta = flatten(update.params) - ref
        return replace(
            update,
            params=unflatten(ref - self.scale * delta, update.params),
            attack=self.tag,
        )


class GaussianNoise(Attack):
    """Model poisoning: add N(0, sigma^2) to every parameter.

    Untargeted noise injection: no direction, just variance. ``sigma`` is absolute
    (parameter units), which makes it directly comparable with the LDP noise scale
    in ``fl/ldp.py`` -- the point of the privacy-vs-Byzantine tension is that an
    honest client under strong LDP and a weak noise attacker are the same vector
    to a defense that only sees updates.
    """

    tag = "gaussian_noise"

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)

    def poison(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate:
        noised = [
            (p + ctx.rng.normal(0.0, self.sigma, size=p.shape)).astype(p.dtype)
            for p in update.params
        ]
        return replace(update, params=noised, attack=self.tag)


class LabelFlip:
    """Data-level poisoning config: flip a fraction of training labels. Consumed
    by the engine before the forward pass, so it deliberately does not implement
    :class:`Attack` (it exposes no ``poison`` method).

    Handshake with ``fl/engine.py``:

    1. The engine builds ``attacks: dict[int, Attack]``. An entry that is a
       ``LabelFlip`` is discriminated by duck typing -- it lacks ``poison`` but
       exposes ``frac`` -- and is never sent through the model-poisoning hook.
    2. The engine applies label flipping *dynamically*, on every training batch of
       every round: a ``frac`` fraction of each batch's labels is moved to a
       uniformly random *different* class (``engine._flip_labels``), drawn from a
       per-(round, client) stream folded from ``seed``. Validation and test labels
       are never touched -- the attacker corrupts what it trains on, not what it is
       evaluated on.
    3. The realized threat model is therefore *dynamic per-batch label noise at
       rate* ``frac``, not a one-time dataset corruption: averaged over rounds it
       approaches symmetric label noise -- a weaker, higher-variance adversary than
       static poisoning. It stays deterministic under a fixed seed.

    The :meth:`flip` method below is a self-contained *static* reference
    implementation (relabel a dataset once, deterministically from
    ``(seed, cid, len(labels))``); it is retained for unit tests and as the basis
    for a static-poisoning variant, and is not on the federated training path.
    """

    tag = "label_flip"

    def __init__(self, frac: float, seed: int = 0) -> None:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"frac must be in [0, 1], got {frac}")
        self.frac = float(frac)
        self.seed = int(seed)

    def flip(self, labels: np.ndarray, n_classes: int = 8, cid: int = 0) -> np.ndarray:
        """Return a copy of ``labels`` with ``frac`` of the entries relabelled.

        Replacements are drawn uniformly from the *other* classes, so every flip
        is a real flip; a flip that lands back on the true class would quietly
        weaken the attack by ``1/n_classes``.
        """
        labels = np.asarray(labels)
        out = labels.copy()
        n = out.size
        k = int(round(self.frac * n))
        if k == 0 or n == 0 or n_classes < 2:
            return out
        rng = np.random.default_rng([self.seed, cid, n])
        idx = rng.choice(n, size=k, replace=False)
        offset = rng.integers(1, n_classes, size=k)
        out[idx] = (out[idx] + offset) % n_classes
        return out


class AdaptiveCosine(Attack):
    """Knows the defense: crafts a malicious update constrained to keep its
    fidelity score below ``target_score`` -- a fixed attacker-side budget, not
    the deployed admission threshold, which it never observes.

    **Knowledge model** (stated honestly -- this is the paper's novelty lever, so
    an inflated attacker would be an inflated result). The attacker knows:

    * the defense's *functional form*: that admission is decided by
      ``1 - cos(delta_i, median(deltas)) <= tau``. It does **not** know tau. The
      attacker never reads, queries or estimates the deployed threshold; it holds
      a fixed attacker-side ``target_score`` (default 0.1, a constructor
      parameter, never derived from the gate) and keeps its own update inside a
      cone of half-angle ``arccos(1 - target_score)`` around the honest
      direction. The adversary is *defense-aware but not tau-aware*: at the
      pre-registered ``tau = 0.5`` it aims at a fifth of the score the gate would
      tolerate, which understates rather than flatters its measured evasion.
    * ``ctx.ref``, the global parameters at the start of the round. Every client
      receives these; they are not privileged.
    * its own honest training result.
    * the same public information one round earlier, which it caches. The
      difference ``ref_t - ref_{t-1}`` is the aggregate step the *server* took,
      the best public estimate of where the honest majority is pushing. Round 0
      has no predecessor, so the attacker falls back on its own honest delta,
      which is itself a draw from the honest distribution.

    The attacker does **not** know: other clients' updates or gradients this
    round, their data, the realised coordinate-wise median, which validators are
    faulty, or the aggregation rule's internal state. It therefore aims at an
    *estimate* of the median direction and can miss; ``safety_margin`` is the
    slack it leaves for that, and the tests measure the realised score rather
    than the targeted one.

    **Construction** (closed form, no rejection sampling). Work in angles, where
    the cosine constraint is a cone half-angle. With ``u`` the unit estimate of
    the honest direction, ``h`` the attacker's honest delta:

        theta_budget = arccos(1 - target_score)      # cone the attacker allots itself
        theta_err    = angle(h, u) / sqrt(K)         # what the estimate may be off by
        theta_use    = max(0, theta_budget - theta_err) * (1 - safety_margin)
        a       = -h                      # locally optimal damage: undo the step
        a_perp  = a - (a.u) u             # the part of it the cone still allows
        w       = a_perp / |a_perp|       # (or the coalition's shared direction)
        delta   = scale * |h| * (cos(theta_use) * u + sin(theta_use) * w)

    ``cos(delta, u) = cos(theta_use)`` identically, because ``u`` and ``w`` are
    orthonormal -- the constraint holds by construction, at any dimension, in one
    evaluation, with no rejection sampling and no query to the defense.

    The two corrections to the naive "sit on the cone boundary" attack are what
    make the realised score, rather than the targeted one, stay under budget:

    * ``theta_err`` -- the defense scores against the *realised* coordinate-wise
      median, not against ``u``. By the triangle inequality on the sphere the
      realised angle is at most ``theta_use + angle(u, median)``, so the attacker
      spends only the part of the budget that survives its own aiming error. It
      estimates that error from the one honest sample it owns: its own delta sits
      ``angle(h, u)`` off the aggregate, and a median over ``K`` such samples
      concentrates as ``1/sqrt(K)`` (``n_peers``, public -- the federation size is
      not a secret).
    * round 0 has no aggregate to aim at, so ``theta_err`` is set to the whole
      budget and ``theta_use`` collapses to zero: **the adaptive attacker spends
      its first round submitting its honest direction** (exactly its honest
      update at the default ``scale``; the magnitude lever, which needs no
      direction estimate, still applies). An attacker that fires blind is
      detected for nothing, and one wasted round out of thirty is cheap -- but it
      is why the attack must be evaluated over at least two rounds.

    ``target_score -> 0`` collapses the attack onto the honest direction (zero
    damage, zero detection) -- the trade-off curve the paper reports.

    ``scale`` is the second, deliberately separate lever: cosine is scale-free, so
    a cosine-only defense cannot see magnitude at all. ``scale > 1`` submits the
    same direction with several times the norm and drags an averaging aggregator
    (geometric median) proportionally further, while the fidelity score does not
    move by a single digit. Defaults to 1.0 -- honest-looking magnitude -- so that
    the headline result is not carried by a trivially clippable norm.

    **Collusion.** With ``collude_seed`` set, every attacker holding the same seed
    derives the same perpendicular direction from ``(collude_seed, ctx.round)``
    instead of from its own ``-h``. The coalition then lands on one point of the
    cone rather than scattering around it, which is what defeats a
    nearest-neighbour rule: Krum scores each vector by its distance to the *dense*
    part of the cloud, and ``f`` mutually identical malicious vectors are their
    own dense part. This is the standard colluding-Byzantine threat model
    (Blanchard et al., NeurIPS 2017, assume exactly this) and it is what the
    contract's ``f = 3/10`` attack fraction means.

    Lineage: cone/inner-product constrained model poisoning -- Baruch, Baruch and
    Goldberg, "A Little Is Enough", NeurIPS 2019; Xie, Koyejo and Gupta, "Fall of
    Empires", UAI 2019; Shejwalkar and Houmansadr, "Manipulating the Byzantine",
    NDSS 2021. What is new here is not the geometry but that the constraint is
    tied to the deployed defense's own published statistic and carried end-to-end
    through consensus into measured accuracy damage.
    """

    tag = "adaptive_cosine"

    def __init__(
        self,
        target_score: float = 0.1,
        scale: float = 1.0,
        safety_margin: float = 0.15,
        collude_seed: int | None = None,
        n_peers: int = 10,
    ) -> None:
        if not 0.0 <= target_score <= 2.0:
            raise ValueError(f"target_score must be in [0, 2], got {target_score}")
        if not 0.0 <= safety_margin < 1.0:
            raise ValueError(f"safety_margin must be in [0, 1), got {safety_margin}")
        if n_peers < 1:
            raise ValueError(f"n_peers must be >= 1, got {n_peers}")
        self.target_score = float(target_score)
        self.scale = float(scale)
        self.safety_margin = float(safety_margin)
        self.collude_seed = collude_seed
        self.n_peers = int(n_peers)
        # Public state carried across rounds: the two most recent global models.
        # Keyed by round so that one shared instance poisoning several clients in
        # the same round does not shift its own history mid-round.
        self._last_round: int | None = None
        self._last_ref: np.ndarray | None = None
        self._prev_ref: np.ndarray | None = None

    def _observe(self, ctx: AttackContext, ref: np.ndarray) -> np.ndarray | None:
        """Cache this round's public global model; return the previous one."""
        if self._last_round is None or ctx.round != self._last_round:
            self._prev_ref = self._last_ref
            self._last_ref = ref
            self._last_round = ctx.round
        return self._prev_ref

    def _direction(self, ctx: AttackContext, u: np.ndarray, a: np.ndarray) -> np.ndarray | None:
        """Unit vector orthogonal to ``u`` along which the damage is applied."""
        if self.collude_seed is not None:
            # Shared across the coalition: same seed, same round -> same vector.
            rng = np.random.default_rng([int(self.collude_seed), int(ctx.round)])
            w = rng.standard_normal(u.size)
            w -= float(w @ u) * u
            # Point it at the damaging half-space when there is one to prefer.
            a_perp = a - float(a @ u) * u
            if float(w @ a_perp) < 0.0:
                w = -w
        else:
            w = a - float(a @ u) * u
        norm = float(np.linalg.norm(w))
        if norm > _NORM_EPS:
            return w / norm
        # Degenerate: the damage direction is collinear with the honest one (or
        # the space is 1-D). Fall back to a random orthogonal direction; if even
        # that fails there is no cone to sit on and the caller gives up.
        w = ctx.rng.standard_normal(u.size)
        w -= float(w @ u) * u
        norm = float(np.linalg.norm(w))
        return w / norm if norm > _NORM_EPS else None

    def poison(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate:
        ref = flatten(ctx.ref)
        h = flatten(update.params) - ref
        prev = self._observe(ctx, ref)

        h_norm = float(np.linalg.norm(h))
        if h_norm <= _NORM_EPS:
            # Nothing was learnt this round, so there is no direction to hide
            # behind and no step to cancel. Submitting the honest (null) update is
            # the correct adaptive choice: anything else is detectable for free.
            return replace(update, params=[p.copy() for p in update.params], attack=self.tag)

        est = ref - prev if prev is not None else h
        est_norm = float(np.linalg.norm(est))
        blind = prev is None or est_norm <= _NORM_EPS
        if blind:  # no observed aggregate step yet: aim at own delta, spend nothing
            est, est_norm = h, h_norm
        u = est / est_norm

        w = self._direction(ctx, u, -h)
        if w is None:
            return replace(update, params=[p.copy() for p in update.params], attack=self.tag)

        theta_budget = float(np.arccos(np.clip(1.0 - self.target_score, -1.0, 1.0)))
        if blind:
            theta_err = theta_budget
        else:
            cos_hu = float(np.clip(float(h @ u) / h_norm, -1.0, 1.0))
            theta_err = float(np.arccos(cos_hu)) / float(np.sqrt(self.n_peers))
        theta = max(0.0, theta_budget - theta_err) * (1.0 - self.safety_margin)

        delta = self.scale * h_norm * (float(np.cos(theta)) * u + float(np.sin(theta)) * w)
        return replace(update, params=unflatten(ref + delta, update.params), attack=self.tag)
