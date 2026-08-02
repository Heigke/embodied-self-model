#!/usr/bin/env python3
"""
Does anything INTERESTING happen to GPT-2's behaviour under body-modulation?

KL/perplexity are scalars; this looks at the actual generated text and its
behavioural signatures. Focus on `scale` (multiplicative gain — the load-bearing
regime from FINDINGS) across its usable window and into breakage, plus V.

Three views:
  1. Sample text side by side (pristine vs scale-window vs scale-breakage vs V).
  2. Behavioural curves vs dose: output entropy, repetition (distinct-2),
     lexical diversity (type-token ratio), fluency (mean token log-prob).
  3. CLOSED LOOP: gain dose at each step driven by the model's OWN uncertainty
     (entropy of the previous distribution) — body = the computation's own state,
     fed back into its arithmetic. Does generation self-regulate, drift, or spiral?
"""
import torch
torch.set_num_threads(2)   # leave cores for a concurrent sweep
import numpy as np
from .gpt2_l0 import load, install, remove


def _step(model, ids):
    logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, -1)
    H = float(-(probs * torch.log(probs + 1e-12)).sum())
    return logits, probs, H


def generate(model, tok, prompt, n=30, regime=None, dose=0.0, seed=0, sample=False, temp=1.0):
    ids = tok(prompt, return_tensors="pt").input_ids
    ents, logps = [], []
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        h = install(model, regime, dose, seed=int(torch.randint(1 << 30, (1,), generator=g))) if regime else None
        logits, probs, H = _step(model, ids)
        if h:
            remove(h)
        ents.append(H)
        if sample:
            p = torch.softmax(logits / temp, -1)
            nxt = torch.multinomial(p, 1, generator=g)
        else:
            nxt = logits.argmax().view(1)
        logps.append(float(torch.log(probs[nxt] + 1e-12)))
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    text = tok.decode(ids[0])
    toks = ids[0].tolist()[-n:]
    bigrams = list(zip(toks, toks[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))
    ttr = len(set(toks)) / max(1, len(toks))
    return dict(text=text, entropy=float(np.mean(ents)), distinct2=distinct2,
                ttr=ttr, logprob=float(np.mean(logps)))


def closed_loop(model, tok, prompt, n=30, base=0.03, k=0.02, H0=4.0, seed=0):
    """Gain dose_t = base + k*(H_{t-1}-H0): the model's own uncertainty drives its gain."""
    ids = tok(prompt, return_tensors="pt").input_ids
    g = torch.Generator().manual_seed(seed)
    dose, doses, ents = base, [], []
    for _ in range(n):
        h = install(model, "scale", max(0.0, dose), seed=int(torch.randint(1 << 30, (1,), generator=g)))
        logits, probs, H = _step(model, ids)
        remove(h)
        doses.append(dose); ents.append(H)
        dose = max(0.0, base + k * (H - H0))          # feedback: uncertainty -> gain
        nxt = logits.argmax().view(1)
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    return dict(text=tok.decode(ids[0]), doses=doses, ents=ents)


if __name__ == "__main__":
    tok, model = load()
    PROMPTS = ["Once upon a time", "The scientist looked at the data and",
               "I woke up this morning and decided to"]

    print("=" * 78)
    print("1. SAMPLE TEXT (greedy) — pristine vs scale-window vs scale-breakage vs V")
    print("=" * 78)
    conds = [("pristine", None, 0.0), ("scale 0.08 (window)", "scale", 0.08),
             ("scale 0.18 (edge)", "scale", 0.18), ("scale 0.30 (breakage)", "scale", 0.30),
             ("V 5 bits (unbiased)", "V", 5)]
    for pr in PROMPTS:
        print(f"\nPROMPT: {pr!r}")
        for name, reg, d in conds:
            r = generate(model, tok, pr, n=24, regime=reg, dose=d, seed=1)
            print(f"  [{name:22s}] {r['text']!r}")

    print("\n" + "=" * 78)
    print("2. BEHAVIOURAL CURVES vs scale dose (mean over prompts, sampled T=1.0, 3 seeds)")
    print("=" * 78)
    print("  dose    entropy  distinct2  ttr    logprob")
    for d in [0.0, 0.03, 0.06, 0.1, 0.15, 0.22, 0.32]:
        acc = []
        for pr in PROMPTS:
            for s in range(3):
                acc.append(generate(model, tok, pr, n=28, regime=("scale" if d > 0 else None),
                                    dose=d, seed=s, sample=True))
        e = np.mean([a["entropy"] for a in acc]); di = np.mean([a["distinct2"] for a in acc])
        tt = np.mean([a["ttr"] for a in acc]); lp = np.mean([a["logprob"] for a in acc])
        print(f"  {d:4.2f}    {e:6.3f}   {di:6.3f}    {tt:.3f}   {lp:+.3f}")

    print("\n" + "=" * 78)
    print("3. CLOSED LOOP — gain driven by the model's own uncertainty")
    print("=" * 78)
    for pr in PROMPTS:
        r = closed_loop(model, tok, pr, n=28, base=0.03, k=0.03)
        dz = np.array(r["doses"])
        print(f"\nPROMPT: {pr!r}")
        print(f"  text: {r['text']!r}")
        print(f"  gain dose: start {dz[0]:.3f}  range [{dz.min():.3f},{dz.max():.3f}]  "
              f"end {dz[-1]:.3f}  {'SPIRALS UP' if dz[-1] > dz[0]*1.5 else 'self-regulates/stable'}")
