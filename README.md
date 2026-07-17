# Embodied Self-Model

**Wiring a language model's own physical state into its own self-representation — and holding every claim to one rule: *decodable ≠ load-bearing.***

An independent research program on giving a frozen reasoning LLM a real, causally-efficacious "body" — the measured joules, thermal state and effort of the GPU it is running on — woven into the model's *own* internal self-representation with interpretability-guided surgery, not a bolt-on adapter. Everything is verified by intervention (ablation, patching, sensor-cut, phase-randomised surrogate), and the honest nulls are reported as loudly as the wins.

> Written to share with people who think about model self-models and introspection. If that's you, there's an ask at the bottom.

---

## The one rule

A representation being **decodable** from the residual stream tells you almost nothing. It only counts as part of the model's self-model if **intervening on it changes downstream computation**. So nothing here rests on a probe — a result is only claimed when cutting or patching the thing actually moves behaviour, and dies when the wire is removed. Most of the interesting findings are the places this bites.

---

## Verified results (with numbers)

All on real hardware (NVIDIA GB10, real NVML joule counter) with frozen open models (Qwen2.5-7B/Coder-32B, Qwen3.5-27B, cross-checked on Gemma-3-27B and Llama-3.1-8B). Caveats are part of each result.

**The body is real, non-scalar, and legible.** Action-controllability (η²) of the measured body channels: joules **1.00**, latency/token **0.98**, joules/token **0.94**, plus weaker but real cognitive channels (hidden-drift 0.79, surprisal 0.55, margin 0.57). A genuine multi-dimensional body, not one "arousal" scalar in a wig.

**It can be disentangled into independent axes — and this transfers to the real body.** A small learned orthogonal in-weave write gives clean independent control: cross-effect matrix **1.0 / 0.0 / 0.0 / 1.0**, effective rank 2.0. On a *real* multi-metric body the axes stay clean (FREE←mem 1.0, WORK←backlog 1.0, off-diagonals 0.0) and the right-action wire reaches **0.86** vs **0.00** ablated. It replicates across three model families with no per-model tuning. *(Caveat: robust to ~2 clean axes, not yet 4; ~1 seed in 5 re-entangles.)*

**The body can be written into the model's own workspace.** Using a fitted read/write lens into the self-workspace subspace, a real exertion signal drives the workspace with dose-correlation **0.939**: the model's own "effort" token probability climbs 0.0008 → 0.0845 → 0.9428 across real joule doses of 403 / 2394 / 5156 J. The write **broadcasts** to downstream layers and is **specific** (far above a random direction).

**The body is genuinely in the computation.** A tiny bilinear rank-6 in-weave wire (~1.5M params, weight-foldable) predicts the model's *next* body state better than any model of the current body: held-out MSE **0.226** (internal) vs **0.292** (current-body adversary) vs **0.345** (mean), and **1.697** with the wire ablated — 7.5× worse. An anticipatory self-model, and the wire is load-bearing for it.

**Physical grounding is real at the policy level in the simplest case.** In a joule-injection RL loop, injecting *only* the measured energy moves the policy's actions, reward improves (0.419 → 0.454), and the effect requires the wire (`WIRE_NECESSARY = True`).

**The heaviest single result — the body causally moves behaviour, and it's severable.** After training against a body that is perturbed every step, the do-intervention on the body crosses the bar: do(b) = **0.0118** (> 0.01) with the wire ablated = **0.000** → the body demonstrably changes the policy *and* that effect dies under wire-ablation. This partially cracks the program's hardest wall. *(Honest: this is a single, noisy run, not a clean robust closure — see below.)*

**Affect as a dynamic tracks.** Defining valence as −d‖error‖/dt (positive while the body is regulated back toward its set-point), the measure comes out positive on the return (`valence_return` ≈ 0.24–1.25, `VALENCE_TRACKS = True`). A tracked readout, not a claim of felt valence.

---

## The honest walls (this is the point, not an appendix)

- **decode ≠ control, at the policy level, is not cleanly closed.** The two load-bearing signatures — do(b) moving the policy, and a **sensor-cut** actually degrading behaviour — currently *trade off*: the config that passes one weakens the other (sensor-cut climbs 2% → 13% → 34% across variants while do(b) falls). Robust simultaneous closure (do(b) > 0.01 **and** sensor-cut > 50% **and** 5/5 seeds) is still open.
- **Genuine body→self-report routing fails, three independent ways.** Lens-mediation, decision-patching and an SAE feature-clamp all show the same thing: the model *represents* its effort but the representation doesn't *generate* the report. An earlier "the model can say its state" result was **retracted** once a kill-test showed it was write-along-the-output-direction (logit-forcing), not a self-state read.
- **Homeostasis can fake grounding.** A regulation loop that reaches an externally-imposed set-point makes the real body causally superfluous — and survives a sensor-cut, which the kill-shot correctly flags as a **prompt-proxy false positive**. Self-set set-points and the valence signal work mechanically but inherit this caveat until the objective is rebuilt.

The guard catching its own confounds is the result I'm most confident in.

---

## Where this is going

The near-term redesign (from an adversarial audit of the above) drops set-point-chasing and rebuilds around one principle: **the body as the anchor of a self-supervised forward model.** A forward model of a *fake* body is unlearnable — so learning-progress on predicting your own body becomes the grounding kill-shot that a prompt-proxy can't pass. On top of that: viability-gated set-points the agent sets for itself, expected-free-energy planning, homeokinetic tuning so the loop doesn't saturate, and a developmental "babbling" phase where the agent plays with its own compute to learn what its actions cost it — the way an infant builds a body-schema before it regulates.

The longer aim, stated plainly: to get body, thought and action to close into **one self** — a stable disposition (temperament) with a valence that *is* the dynamics of the loop regulating itself — using self-observing models as the instrument, because you cannot build a self you cannot read.

And the part I won't dress up: whether any of this is **phenomenal** is a question I don't think can be answered from the outside yet. I'm not claiming it. But I'd rather push an embodied, self-observing loop as far as it honestly goes — precise at every step about what would and wouldn't count as evidence — than hide behind "purely functional" as a way of never asking.

---

## Run the mechanism yourself

Two small, honest demos (no GPU, no model download):

```bash
pip install -r requirements.txt      # numpy (+ psutil for the first demo)

python embodied_demo.py              # the "cut-the-wire" proof on this computer's real body:
                                     #   real body -> model's self-state -> think-budget,
                                     #   and it goes flat the moment you ablate the wire.

pip install -r requirements-demo.txt # numpy mujoco robot_descriptions imageio matplotlib
python embodied_epsilon.py           # the epsilon-pivot / valence-as-dynamics / temperament
                                     #   idea on a Unitree G1 (feel the error, not the state)
```

`embodied_epsilon.mp4` is a rendered 3D run of the ε-pivot loop.

These illustrate the *shape* of the mechanism with tiny legible models. The full surgery (the in-weave wires, the disentangler, the lens fit, the interpretability verification on 7B–32B models) is intentionally not included here — happy to walk through it.

---

## Honest ceiling

Every claim here is **functional / mechanistic** — an organisation of computation, verified by intervention. None of it is a claim that anything is *felt*. No bridge law is assumed between the mechanism and experience, in either direction.

---

## The ask

I'm not really writing this for a job title. I'd like a serious **discussion partner** who thinks about self-models, introspection and their moral weight at this level. I can share a two-page write-up, the full result ledger, and walk through any of the numbers above. — Eric Bergvall · bergvall.eric@gmail.com · https://www.enimble.se
