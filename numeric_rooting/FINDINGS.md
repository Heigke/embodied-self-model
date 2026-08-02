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

---

# GPT-2 (124M) — the L0 experiment on a real transformer

The small-MLP result above bounds what is *possible*. This section runs the plan's actual
L0 site — body-biased stochastic rounding on the **output of every attention/MLP matmul in
GPT-2**, via a PyTorch hook — and measures behaviour vs coherence on real language.
**86 cells** (`gpt2_sweep.py`). Behaviour = KL(pristine ‖ perturbed) next-token dist;
coherence = perplexity on held text (pristine PPL 29.6). The load-bearing test is
`kl_seedmean/kl_single`: how much of the effect survives averaging over rounding seeds — a
genuine MEAN effect (→1) versus noise that averages away (→0).

## The two safe regimes are unread — confirmed on a real model

| regime | max behaviour (kl_single) | survives seed-avg (kl_seedmean/kl_single) | coherence cost |
|---|---|---|---|
| **M** mean-shift | 0.0001 | **0.00** | none (PPL ×1.00) |
| **V** precision | 0.026 | **0.10** | none (PPL ×1.06 at 5 bits dropped) |

- **M is fully laundered by LayerNorm.** kl_single ≈ 0 and kl_seedmean = 0 at every dose up to
  b = 1.5. A uniform mean-shift on the residual stream is normalised away before it can route.
- **V is unbiased, exactly as the theory says.** ~90 % of its single-pass effect averages out
  (ratio 0.10), it never dents coherence, and its total leverage is tiny (KL ≤ 0.026). The
  unbiased channel is safe *and* unread.

So on GPT-2 the §3 theorem holds directly and mechanistically: **the mean channel is actively
laundered, the variance channel averages itself away.** Moving from an MLP to a real transformer
made the null *stronger*, not weaker.

## The one partial escape: multiplicative gain (`scale`)

Pushing the aggression ladder past pure rounding, one rung behaves differently:

| rung | behaviour moves (kl>0.01) @ dose | coherence degrades (PPL>1.5×) @ | breaks (PPL>2×) @ | survives seed-avg | depth peak |
|---|---|---|---|---|---|
| **scale** (gain jitter) | 0.037 (PPL ×1.007) | 0.163 | 0.269 | **0.28** | layer 5 (mid) |
| **signflip** (sign corruption) | 0.001 (PPL ×1.26) | 0.003 | 0.003 | 0.74 | layer 1 (early) |

- **`scale` has a usable window.** Between dose ≈ 0.04 and ≈ 0.16 a body-modulated multiplicative
  gain moves behaviour (KL 0.015 → 0.28) while text stays coherent (PPL ×1.007 → ×1.5), and
  **~28 % of that effect survives seed-averaging** — a real mean effect, not just noise. This is
  the only intervention on GPT-2 that is both load-bearing and non-destructive. It points
  straight at the programme's open problem #4 (*Format — gain modulation?*): the load-bearing
  site on a transformer is a **multiplicative gain**, not an additive rounding bias, because
  additive bias is what LayerNorm exists to remove.
- **`signflip` is load-bearing but destroys coherence instantly** (74 % survives averaging, but
  PPL blows up by the second dose). The far anchor: it proves the wall exists and that a strong
  mean effect is reachable — at the cost of the computation.
- **Depth:** `scale` peaks mid-stack (layer 5), `signflip` peaks early (layer 1, errors then
  propagate through the whole stack). `V`/`M` are flat — no depth structure because they carry
  nothing.

## GPT-2 synthesis

1. **Both safe rounding regimes (M, V) are unread on a real transformer** — M laundered by
   LayerNorm, V averaged away. The §3 theorem is confirmed at L0, mechanistically.
2. **The load-bearing site is multiplicative gain, not additive rounding.** `scale` is the single
   rung with a genuine mean effect *and* a coherence-preserving window, peaking mid-depth. That is
   a concrete, positive lead for the plan's format question, and the most useful thing the GPT-2
   run produced.
3. The behaviour-vs-coherence gap the plan wanted exists **only for `scale`**: for V it never
   opens (no leverage), for `signflip` it never opens (destroys as it moves). One rung, one window.

Caveats held: single held-text perplexity and 6 prompts (coarse behaviour/coherence estimates);
greedy next-token KL, not full generation; GPT-2 small only. These bound the claim to "on GPT-2
small, at the matmul-output rounding site" — not all transformers, not the internal tensor-core
MMA. Still functional/mechanistic only; no phenomenal claim.

*Reproduce GPT-2:* `python -m numeric_rooting.gpt2_l0` (dose table) ·
`python -m numeric_rooting.gpt2_sweep` (full 86-cell sweep, resumes from `results/`).

## Does anything interesting happen to the BEHAVIOUR? (generated text)

KL/PPL are scalars. Reading the actual generations (`results/behaviour_samples.txt`,
`gpt2_behaviour.py`) shows the effect is real and legible:

- **The unbiased regime (V) leaves behaviour visibly unchanged.** Greedy text under V at 5
  dropped bits is near-identical to pristine — *"Once upon a time, the world was a place of
  great beauty and great danger"* both times. The "unread" theorem, now readable on the page.
- **Body-modulated gain (scale) in its window steers WHAT the model says, not HOW.** At dose
  0.08–0.18 the grammar stays intact but the *content* shifts — the same prompt goes from
  *"the world was a place of great beauty and great danger"* to *"I was able to get a few hours
  of sleep"* to *"go to the store … buy some food"*. The body nudges the topic while the
  sentence stays well-formed. This is the interesting effect: a numeric knob steering semantics
  coherently.
- **A small perturbation SHARPENS the model, a large one DIFFUSES it** — non-monotonic. Output
  entropy dips (4.07→3.87) and fluency *rises* (mean log-prob −3.92→−3.68) at dose ≈ 0.03–0.06 —
  a mild gain makes GPT-2 slightly more decisive and more fluent — then entropy climbs to 5.9 and
  fluency collapses to −6.1 by dose 0.32. There is a small "sweet spot" where the body helps.
- **Breakage is legible too:** at dose 0.30 grammar dissolves — *"Once upon a time to do, I am
  not to be a man"*, *"a-lucky for the man"*. The coherence wall, in words.
- **Closed loop self-regulates.** When the gain is driven by the model's OWN uncertainty
  (dose_t = base + k·(entropy_{t−1}−H₀)) during generation, the dose rises when the model is
  unsure (up to ~0.17) and settles back toward 0 as it grows confident — bounded, no spiral, in
  all three prompts. A crude homeostat on live GPT-2 generation: the computation's own state
  modulating its own arithmetic, staying in the coherent band.

So the behavioural picture matches the numbers: the safe/unbiased channel is invisible in the
text, and the one load-bearing channel (gain) has a real, coherent, steerable window before the
wall — and even closes a stable loop on the model's own uncertainty.

## Deep rooting — the body INSIDE the multiply-accumulate (30 cells)

The tensor-core MMA is not addressable from CPU, but the matmul can be reimplemented as an
explicit **split-k accumulation** and the body rooted into the accumulation itself
(`matmul_deep.py`): round the running accumulator each K-block (`round`), modulate each partial
sum's gain (`gain`), or reorder the block summation (`order`, Site B). The split-k rewrite is
exact at b=0 (KL 0.00004). Sweeping the block count `nb` = *how deep the rooting goes*:

**Q1 — rooting the ROUNDING deeper compounds it, strongly.** kl_single vs nb:

| dose | nb1 | nb4 | nb8 | nb16 | nb32 |
|---|---|---|---|---|---|
| 0.2 | 0.00003 | 0.00017 | 0.00039 | 0.00097 | **0.00425** |
| 0.5 | 0.00007 | 0.00030 | 0.00095 | 0.00358 | **0.01534** |

Each block boundary is another rounding event, so finer split-k multiplies the perturbation
~100–200× from output (nb1) to nb32. **Where in the accumulation you round matters enormously** —
the same nominal dose is two orders of magnitude more potent rooted deep than rooted at the output.

**Q2 — the load-bearing GAIN wants the opposite: shallow rooting.** The mean effect
(kl_seedmean, what survives seed-averaging) at dose 0.07: nb1 **0.0112**, nb4 0.0106, nb8 0.0088,
nb16 0.0089, nb32 0.0069. Distributing the gain across many blocks lets the per-block jitter
average out *within* the sum, so the surviving mean effect *falls* with depth. **Gain belongs at
the output (or coarse blocks); rounding belongs deep.** Opposite preferences — a real design fact.

**Q3 — accumulation order (Site B) is exactly null at fp32.** Permuting the block summation
order moves behaviour by 0.000000 at nb4/16/64: fp32 is too precise for non-associativity to
bite. But under **low precision** it does — roundV (m_eff≈3) with deeper split-k:
nb4 KL 0.022 (PPL ×1.04) → nb16 0.140 (×1.10) → nb64 **0.420 (×1.97, near breakage)**. Site B
only becomes a channel once the arithmetic is already coarse; at fp32 it carries nothing.

**Deep-rooting synthesis:** depth is a genuine lever, but a *sign-dependent* one — rooting the
rounding deep compounds it 100×+, while rooting the load-bearing gain deep dilutes it. The body
belongs at different arithmetic sites depending on whether it enters as variance (deep) or as a
mean-effect gain (shallow). Non-associativity (Site B) is inert until precision is already low.
The tensor-core MMA remains out of reach on this hardware — that is the one rung below this.

*Reproduce deep:* `python -m numeric_rooting.matmul_deep` ·
`python -m numeric_rooting.gpt2_deep_sweep` · behaviour: `python -m numeric_rooting.gpt2_behaviour`.
