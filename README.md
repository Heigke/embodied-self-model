# Embodied Self-Model

Feeding a language model **its own measured hardware state** — joules, temperature, throttle, pressure — and asking whether it becomes *load-bearing* in the model's own self-representation.

One rule governs everything: **decodable ≠ load-bearing.** A probe recovering a representation proves nothing. It counts only if intervening on it changes downstream computation. Nothing here rests on a correlation.

The most useful part of this repo is where that rule bit us → **[NOTES_how_we_fooled_ourselves.md](NOTES_how_we_fooled_ourselves.md)**

Independent research, one 119 GB GB10 box, open models (Qwen 7B–35B).

---

## Holds

| Result | Number | Control it cleared |
|---|---|---|
| Self-report tracks **real chip temperature** | 0.82 | random direction 0.20 |
| Forward model learns the action→Δbody law | z = 6.77 | shuffled body can't learn |
| Per-organ selectivity (disk→nvme, gpu→energy) | R² 0.95 / 0.98 | 4/4 seeds, z_med 8.60 |
| Residual-stream write flips the policy | ΔP +0.83 | argmax flip 8/8 |
| Forget-free real-weight consolidation | KL 0.39 | LoRA 7.73 |
| Self-state survives a silent gap | 0.343 | scrub kills it |
| Decodable self-locus (not the token "I") | d = 4.59 | permutation z = 23 |
| Body-blind agent **dies**; body-aware rests and solves | 1/6 vs solved | same environment |

## Does not hold

- **Decodable, not controllable.** Steering the self-locus doesn't clear a random band (z = 2.84). DAS interchange finds **no causal self-subspace** — IIA = 0.0 at every layer and rank.
- **Two keys, two locks.** The additive coupling that moves behaviour (0.95) can't be safely folded into weights; the foldable form that forgets nothing can't carry the body (0.037).
- **Represent but don't route.** Lens mediation, decision patching and an SAE clamp all agree: the state is represented, not routed into the report.
- **We retracted our own "sayable body" result** — a kill-test showed it was logit forcing.
- **Homeostasis can be a prompt-proxy.** Regulation that survives cutting the sensor was regulating a *predicted* body.
- **Introspection null at 35B** — but capability-gated, so ambiguous. 70B doesn't fit the box.

## Now (2026-07-27)

Backbone **thawed** at unit-selected sites — it was frozen for seven months, which made every "it doesn't read the body" result uninterpretable. Write-channel positive control now passes at **z = 26.7** (was 1.14); **every null recorded before that repair is void.** Body being widened: the old vector spanned ~2.57 directions with `gpu_temp ~ gpu_power` at **r = 0.992** — five views of one heat curve, not five organs.

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

## Ceiling

Functional and mechanistic only. No claim anything is *felt*; no bridge law assumed either way. Several consciousness indicators are pass-by-construction if you design to them — and the non-gameable ones are exactly the ones we currently fail.

**Eric Bergvall** · bergvall.eric@gmail.com · [enimble.se](https://www.enimble.se)
Critique welcome, especially the kind that kills a result.
