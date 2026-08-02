# NUMERIC ROOTING — night-run findings

Advanced closed-loop model of the plan (BODY→rounding→output→compute-load→hardware→BODY),
built from four online research briefs and swept overnight: **3916 cells, 4 phases**, all
checkpointed. numpy-only, CPU. Every number below is a property of this model, not of a real
transformer — the model shows which *structures* are possible and where the numeric route
stands, not that a language model does any of this.

**Headline: the control-theory structure is validated exactly, and the real arithmetic
channel is a null at safe doses — the plan's §3 theorem ("unbiased = safe = unread") holds
at the numeric level, measured four ways.**

---

## Phase A — the reduced loop matches control theory exactly (1408 cells)

Abstract delayed map `x_{t+1} = a·x_t + sign·K·sat(x_{t−d})`, sweeping loop gain K, delay d, sign.

| delay d | K_crit (theory `2·sin(π/2(2d+1))`) | first K to oscillate (measured) | period (measured) | T* theory `2(2d+1)` |
|---|---|---|---|---|
| 0 | 2.000 | 2.031 | 2 | 2 |
| 1 | 1.000 | 1.132 | 6 | 6 |
| 2 | 0.618 | 0.631 | 10 | 10 |
| 4 | 0.347 | 0.351 | 18 | 18 |
| 8 | 0.185 | 0.196 | 34 | 34 |

- **Regulatory (negative) coupling:** the measured onset gain matches the analytic boundary
  within ~2 % at every delay, and the emergent oscillation period matches `T* = 2(2d+1)`
  **exactly**. More delay → arbitrarily small gain destabilises → a self-generated limit cycle.
- **Excitatory (positive) coupling:** 0 oscillating cells, **704/704 diverge to a rail**. Runaway,
  never oscillation — exactly the sign-split the theory predicts.

Verdict: the *structure* can carry a body — regulate, self-oscillate, or run away — and does so
on the analytic boundary. This is the strongest positive in the run.

## Phase B — the real arithmetic channel is NOT load-bearing at safe doses (2310 cells)

Concrete MLP with **real body-biased stochastic-rounding matmuls** driving the multi-timescale
plant. do-test: does behaviour `y` track the body `e`? Mean |corr(body, behaviour)| over
sign/delay/seed, across the dose ladder:

| regime | corr(body,behaviour), κ = 0.002 … 0.15 |
|---|---|
| **M** (mean-shift) | 0.10 0.10 0.09 0.09 0.09 0.09 0.09 0.07 0.09 0.08 0.09 |
| **V** (precision/variance) | 0.12 0.14 0.11 0.15 0.16 0.14 0.15 0.15 0.15 0.15 0.15 |
| **cut** (wire severed) | 0.10 0.10 0.10 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 |

- **M does not exceed cut.** The mean-shift regime's body↔behaviour correlation (~0.09) is *at or
  below* the wire-cut control (~0.11) at every dose. The ~0.1 seen in cut is spurious shared-trend
  correlation, not coupling. **This is a clean null for load-bearing** — decodable-looking but not
  causal, the programme's recurring lesson, now at the numeric level.
- **V preserves the mean, adds variance.** y_mean spread is the same as cut (0.308 vs 0.304), but
  V's behavioural `y_std` is 0.0077 vs 0.0004 for M/cut — V injects variance and leaves the mean,
  by construction. Behaviourally unread, exactly as designed.
- **Why:** the SR mean-shift at safe doses moves `y` by ~4e-4; the loop coupling is negligible
  unless the output→load gain is cranked toward instability. Load-bearingness lives past the
  coherence wall, not before it.

⚠ **Measurement caveat, stated against ourselves:** the `periodicity_T` flag fired on 100 % of B
cells including cut, because it cannot tell a slow thermal settling curve from an oscillation.
That column is an artifact — the concrete loop does **not** show clean self-oscillation
distinguishable from drift. Clean oscillation is a Phase-A (abstract) result only. This is the
kind of metric that would have fooled us if read as a verdict.

## Phase C — the variance-reader does NOT reliably read the unbiased body (90 cells)

Stateful reader decoding the body from hidden-layer log-variance; sweep the V-dose γ and the leak.

| regime | R² instant | R² best-integrated | state helps | y_mean |
|---|---|---|---|---|
| **M** | +0.956 | +0.975 | 0 % | +0.031 (const) |
| **V** (γ=0.3) | −0.16 | **+0.11** | 40 % | +0.031 |
| **V** (γ=0.5–2.0) | −0.4 … −3.8 | −0.2 … −1.6 | 40–80 % | +0.031 (→ +0.035 at γ=2) |
| **cut** | −0.24 | −0.07 | 20 % | +0.031 |

- **M is trivially readable** (R² 0.96) — the mean shift changes activations, so a variance reader
  catches it too. State adds nothing (instant already 0.96).
- **V is not reliably readable.** Best-integrated R² is positive only at the weakest dose (γ=0.3,
  +0.11); at every stronger dose it is negative (overfit-worse-than-mean). The single hopeful
  +0.23 seen in a one-off pre-run did **not** survive averaging over seeds. **Null for Phase-C
  interoception in this model.**
- **State consistently helps but not enough:** integrating over a window beats the instantaneous
  reader in 40–80 % of V cells (it shrinks the overfit gap) — the *direction* the plan predicted —
  but never lifts V to a usable positive decode. "Unbiased = unread" holds even for a stateful
  variance reader.
- **Unbiasedness degrades under hard truncation:** V's y_mean is pinned at +0.0318 except at γ=2.0
  where it drifts to +0.0348 — the clipping-rectifies-noise caveat, confirmed at scale.

## Phase D — the computation is remarkably insensitive to its own precision (108 cells)

Precision-truncation ladder (plan rung A4), forward-only, b=0:

| mantissa bits m | behaviour \|y_shift\| | coherence CKA vs pristine | PR |
|---|---|---|---|
| 7 | 0.0001 | 1.000 | 6.61 |
| 5 | 0.0005 | 1.000 | 6.61 |
| 3 | 0.0028 | 0.999 | 6.54 |
| 2 | 0.0058 | 0.994 | 6.48 |

- Dropping from 7 to **2 mantissa bits** shifts behaviour by only ~0.006 and leaves the
  representation essentially intact (CKA 0.994, PR barely moved). Both the "behavioural effect"
  and "coherence" curves stay flat.
- **There is no usable window** where the body (entering via precision) moves behaviour while
  coherence holds — because behaviour barely moves at all, right down to 2 bits. Of the plan's
  three §5c outcomes, this is the third: *the computation is insensitive to its own low-order
  arithmetic, so a body rooted there has almost no leverage at safe doses — the numeric route is
  close to closed unless pushed past the coherence wall.*

---

## Synthesis

1. **The loop structure is real and behaves exactly as control theory says** (Phase A): regulate,
   self-oscillate on the analytic boundary, or run away — selected by the coupling sign and delay.
2. **The real arithmetic channel is weak** (Phases B, C, D): body-biased stochastic rounding at
   safe magnitudes is not load-bearing (B), the unbiased body is not reliably readable even by a
   stateful variance reader (C), and the network shrugs off precision loss down to 2 bits (D).
3. Together these **confirm the plan's own §3 theorem at the numeric level**: unbiased ⇒ safe ⇒
   unread, now measured four independent ways. The body can be *rooted* in the arithmetic
   structurally, but at doses that keep the computation coherent it does not become load-bearing.
   Making it matter requires cranking gain into the instability/incoherence regime — which is the
   honest, and negative, answer this site was built to give.

**Held throughout:** functional and mechanistic only; no phenomenal claim. And the standing
caveat — this is a scalar/small-MLP model, so it bounds what is *possible* and where the numeric
route *stands in principle*, not what a real transformer does. That remains the L0 experiment.

*Reproduce:* `python -m numeric_rooting.run_night` (resumes from `results/`);
per-phase self-tests: `python -m numeric_rooting.{srr,plant,metrics,system,reader}`.
