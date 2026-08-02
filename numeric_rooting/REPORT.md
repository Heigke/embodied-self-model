# Numeric Rooting — short report

**Question.** Can coupling a physical "body" signal *into the arithmetic itself* (the rounding /
gain decision inside a matmul) be **load-bearing** — change what a network does — and give any
**advantage** (a usable temperament, a task edge)? The rule throughout: the body must not be
*noise going in*, but a signal the model can act *with*. Ceiling held: functional/mechanistic
only, no phenomenal claim.

**Two testbeds.** (1) a scalar/small-MLP **closed loop** where the compute the model does drives a
simulated hardware plant whose state biases the arithmetic (numpy); (2) **GPT-2 small (124M)**,
where body-modulation is injected into real attention/MLP matmuls via PyTorch hooks / a split-k
reimplementation, measured on real language. Everything checkpointed and reproducible.

---

## What we tested

| # | Test | Where |
|---|---|---|
| 1 | Closed-loop control theory (stability, delay-oscillation, sign) | MLP, 1408 cells |
| 2 | Load-bearing of body-biased rounding (do-test), dose to destruction | MLP, 2310 cells |
| 3 | Interoceptive reader: decode body from activation variance | MLP, 90 cells |
| 4 | Aggression ladder: behaviour vs coherence | MLP + GPT-2 |
| 5 | GPT-2 L0: body-biased rounding on real matmul outputs (M/V) | GPT-2, 86 cells |
| 6 | Deep rooting: body inside the split-k accumulation (round/gain/order) | GPT-2, 30 cells |
| 7 | Behaviour in generated text; closed loop on the model's own uncertainty | GPT-2 |
| 8 | Exchange study: body vs matched noise vs strict (personality, task) | GPT-2 |
| 9 | Signal-lifting: recover a coherent body from noise-like fluctuation | GPT-2 |

## What we found

**Control-theory structure is real and exact (1).** The reduced loop's oscillation onset matches
the analytic boundary `K_crit(d)=2·sin(π/2(2d+1))` within ~2 %, and the emergent period matches
`T*=2(2d+1)` exactly. Regulatory coupling + sensor delay → self-oscillation; excitatory coupling →
runaway (704/704). The loop *can* carry a body.

**But safe arithmetic is unread — confirmed four ways (2,3,5).** Body-biased **rounding** at
coherence-preserving doses is **not load-bearing**: in the MLP loop, mean-shift (M) correlates with
behaviour no more than the wire-cut control; on GPT-2, **M is fully laundered by LayerNorm**
(KL=0 per-pass *and* seed-averaged) and **V (precision) is unbiased** (single-pass KL 0.011 →
0.001 after seed-averaging). A stateful variance-reader in the MLP could not reliably decode the
unbiased body either. This is the plan's theorem — *unbiased = safe = unread* — holding on a real
transformer.

**The load-bearing site is multiplicative GAIN, not additive rounding (5,6).** On GPT-2 only the
`scale` (gain) rung has both a mean effect that survives averaging (~28 %) **and** a usable window
(dose ≈ 0.04–0.16) where behaviour moves while text stays coherent; it peaks mid-depth (layer 5).
Additive bias is exactly what LayerNorm removes. Deep-rooting is a *sign-dependent* lever: rooting
**rounding** deep in the accumulation compounds it 100–200×, but rooting **gain** deep *dilutes* it
(per-block jitter averages within the sum). Accumulation order (Site B) is null at fp32, biting
only under low precision.

**The effect is legible in behaviour (7).** V leaves generated text near-identical to pristine
(unread, on the page). Gain in its window steers *what* GPT-2 says while grammar holds; a small
gain even sharpens it (lower entropy, higher fluency); past the window grammar dissolves. A closed
loop driving gain from the model's own uncertainty self-regulates (bounded, no spiral).

**The exchange — what giving up bit-exactness buys (8,9):**
- **A reproducible, orderable temperament.** Same body → identical output; matched noise → different
  every run. Four bodies are separable; dose is an orderable coherence dial. Noise gives none of this.
- **A modest task edge when the body senses the computation.** Entropy-driven gain breaks repetition
  more efficiently than blind noise (more diversity per unit coherence lost).
- **⭐ A genuinely readable channel.** An unbiased, mean-preserved, per-unit-noise-like fluctuation
  whose variance is modulated by a **coherent** signal b(t) is **recoverable** by population-variance
  averaging + temporal integration: **R²=0.62** (MI 0.73). Matched **incoherent** noise of identical
  magnitude gives **R²=0.00**. A √N signature (R² 1-unit 0.41 → 768-unit 0.62; incoherent flat)
  confirms population coding. Structure, not magnitude, is the entire difference.

## Bottom line

Coupling a body into strict rounding, at doses that keep the computation coherent, is **not**
load-bearing — the mean channel is laundered, the variance channel averages away. The escape is
**gain modulation**, which has a real coherent window, and — decisively — a **coherent** body is
**not noise**: it survives as a recoverable signal in the population variance (interoception in the
correct sense), while an incoherent body of the same magnitude is measurably just noise. So the
value of giving up bit-exact arithmetic is real but conditional: you gain a reproducible temperament
dial and a readable interoceptive channel **iff you supply coherence**; the cost is coherence-loss
with dose, which sets the usable window.

**Open next step:** the model here is *perturbed* by the body, not *trained to use* it. The natural
continuation is to train a small readout (or the backward pass) so GPT-2 learns to read its own
population variance — turning the recoverable signal (9) into behaviour the model acts on.

*Full detail and numbers: `FINDINGS.md`. All code and checkpointed results under `numeric_rooting/`.*
