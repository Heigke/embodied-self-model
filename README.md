# Embodied Self-Model

Feeding a language model **its own measured hardware state** — joules, temperature, clock, contention, memory pressure — and asking whether it becomes *load-bearing* in the model's own self-representation, in real weights, while it runs.

One rule governs everything: **decodable ≠ load-bearing.** A probe recovering a representation proves nothing. It counts only if intervening on it changes downstream computation. Nothing here rests on a correlation.

**Most results here are nulls.** That is the point, and the most useful file in the repo is the one where the rule bit us → **[NOTES_how_we_fooled_ourselves.md](NOTES_how_we_fooled_ourselves.md)** — thirteen results we killed ourselves, with the arithmetic.

Independent research. Evenings, borrowed workstations, one 119 GB GB10 box, open models (Qwen 7B–35B).

---

## The headline result

We re-implemented the **Jacobian lens** ([Gurnee, Sofroniew, Lindsey et al., July 2026](https://transformer-circuits.pub/2026/workspace/index.html)) and pointed it at a signal that paper explicitly places out of scope — the model's own *physical implementation*.

> **The body arrives and does not enter the verbalizable workspace.**
> Workspace loading **z = −0.14**, **0 of 23 layers** above z = 2.
> Positive control on the same rig reads *Italy* and *boot* correctly, so the instrument is not silent.

The paper states its results "are not relevant to assessing consciousness according to [substrate-based] theories" because they concern computation rather than physical implementation. Both invited commentaries then named a body as the gap — Dehaene & Naccache on "its lack of a body… capable of emitting pleasure or pain signals", and Eleos citing Seth on interoception. This repo is an attempt to build that arm and measure it rather than argue about it.

**Caveat we hold against ourselves:** every one of our self-observation nulls so far measured an *untrained* write. Whether a null there is informative at all is currently an open question, and it is the one we would most like an outside answer to.

---

---

## Also holds

| Result | Number | Control it cleared |
|---|---|---|
| Self-report tracks **real chip temperature** | 0.82 | random direction 0.20 |
| Forward model learns the action→Δbody law | z = 6.77 | shuffled body can't learn |
| Per-organ selectivity (disk→nvme, gpu→energy) | R² 0.95 / 0.98 | 4/4 seeds, z_med 8.60 |
| Forget-free real-weight consolidation | KL 0.39 | LoRA 7.73 |
| Self-state survives a silent gap | 0.343 | scrub kills it |
| Decodable self-locus (not the token "I") | d = 4.59 | permutation z = 23 |
| Body-blind agent **dies**; body-aware rests and solves | 1/6 vs solved | same environment |
| Sense organ sees its world on an **empty** machine | AUC 0.854 vs null 0.508 | same code at 40.9 W: 0.550 vs 0.540 |
| **CKA cannot be a recoverability gate** | flat to 4 d.p. while accuracy falls 0.944→0.778 | KL moves 160× in the same window |

## Does not hold

- **Decodable, not controllable.** Steering the self-locus doesn't clear a random band (z = 2.84). DAS interchange finds **no causal self-subspace** — IIA = 0.0 at every layer and rank.
- **Two keys, two locks.** The additive coupling that moves behaviour (0.95) can't be safely folded into weights; the foldable form that forgets nothing can't carry the body (0.037).
- **Represent but don't route.** Lens mediation, decision patching and an SAE clamp all agree: the state is represented, not routed into the report.
- **We retracted our own "sayable body" result** — a kill-test showed it was logit forcing.
- **Homeostasis can be a prompt-proxy.** Regulation that survives cutting the sensor was regulating a *predicted* body.
- **Introspection null at 35B** — but capability-gated, so ambiguous. 70B doesn't fit the box. Binary yes/no detection is now known to be confounded by a global affirmative shift (independently: net signal −0.01 ± 0.03 after control subtraction); forced-choice-from-N and within-trial magnitude comparison survive.
- **The body cannot be sensed under our own footprint.** A co-tenant occupying 61 % of wall clock is invisible at 40.9 W (AUC 0.550 vs null 0.540) and clearly visible at 5.5 W (0.854). Aggregate telemetry is a scalar sum; attribution is the missing equation.

## Now (2026-07-30)

Backbone **thawed** at unit-selected sites — it was frozen for seven months, which made every "it doesn't read the body" result uninterpretable.

Three results were struck this week, two of them by an outside reviewer who read the code and ran its own controls:

- **An "ignition" measurement was an algebraic identity.** Injection scale 575 × cone fraction 0.9326 = 536.2; measured 537.3. A no-network synthetic reproduces it with relative range 0.000000. The pre-registered bar was also unreachable: a *planted* 50 % all-or-none amplification scores FAIL on our own rig. Anthropic had already run the experiment on the right axis (mixture between competing concepts, not amplitude) and found sharp bimodal commitment — the opposite conclusion.
- **A "real hardware wired to decisions" result was a two-point lookup table.** The normaliser maps any two body readings to ±the same vector, so the model never saw more than one bit; evaluation was on the training prompts verbatim; and the joule channel sat on the decision boundary carrying nothing.
- **A previously reported write-channel positive control (z = 26.7) and a ΔP = +0.83 policy flip are withdrawn** — the first failed a specificity control, the second is not supported by the artifact.

**Rule added:** suspiciously good replication is evidence of a tautology. Two runs agreeing to 1.4 % across changed prompts, grids and direction construction should trigger a synthetic no-network control, not a bank.

## Why

The reason to root a body in at all: a system with real limits, real costs and a real edge against a chaotic environment has something to explore *from*, and something to be careful *about*. If any of that is achievable it should show up as measurable consequences in the model's own computation — not as language about feelings. That is the whole bet, and it is currently unproven.

**Ceiling, held throughout:** functional and mechanistic. No claim anything is felt, no bridge law assumed in either direction.

## Open problems

1. **Write-to-route.** Reading is mature, writing is blunt. Is there a *surgical* intervention that makes a non-lexical signal load-bearing without wrecking capability?
2. **Safe *and* efficacious.** One mechanism that both folds without forgetting and gets read — or is the trade-off structural?
3. **Non-lexical signals in a verbalizable workspace.** If workspace directions are vocabulary-indexed, what would count as evidence a body signal is in there *as itself*?
4. **Format.** Continuous projection measures *worse than no body at all*. Tokens? Gain modulation? Forward-model modules?
5. **Separation statistics.** Effective rank on a small matrix is not evidence of separation (see notes). What is?

## Run it

```bash
pip install -r requirements.txt      # numpy, psutil
python embodied_demo.py              # cut-the-wire proof on this machine's real body
```

```bash
pip install -r requirements-demo.txt # + mujoco
python embodied_epsilon.py           # feed the body as prediction ERROR, not state (G1 sim)
```

`embodied_epsilon.mp4` — rendered run of that loop.

---

**Eric Bergvall** · Stockholm, Sweden · bergvall.eric@gmail.com · [enimble.se](https://www.enimble.se)

Critique welcome, especially the kind that kills a result. Three of the thirteen entries in the notes were found by asking someone outside the project to read the code and try to break it — that is the cheapest habit on the list and the one we adopted last.
