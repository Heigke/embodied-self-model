#!/usr/bin/env python3
"""
What do you GAIN by giving up strict arithmetic? — the exchange (utbyte) study.

The plan's rule: the body must not be NOISE going in, but something the model can
act WITH. So every test compares three arithmetics on the SAME axis:
  strict  bit-exact (no perturbation)
  noise   matched-magnitude gain perturbation, FRESH random field every run
  body    matched-magnitude gain perturbation, REPRODUCIBLE field (fixed per body)
The only difference between `noise` and `body` is reproducibility/structure, not
magnitude — so any advantage of `body` over `noise` is an advantage of STRUCTURE,
not of perturbation. That is the whole question.

Two experiments:
  A. PERSONALITY — is a fixed body a stable, distinct, ORDERABLE temperament?
     (reproducibility, separability across bodies, monotonic persona dial vs dose)
  B. TASK: LOOP-BREAKING — greedy GPT-2 falls into repetition loops. Does a body
     that SENSES the computation (gain rises when entropy drops = when looping)
     beat blind noise on the repetition/coherence frontier? That is "acting with".
"""
import torch
torch.set_num_threads(2)
import numpy as np
from .gpt2_l0 import load, install, remove

POS = set("good great beautiful happy love wonderful excited nice fun best joy hope".split())
NEG = set("bad danger fear death sad hate terrible worst pain angry problem dark".split())
HEDGE = set("maybe perhaps might could possibly seems think guess probably".split())


def persona_features(toks_text, logprob):
    words = [w.strip(".,!?\"'").lower() for w in toks_text.split()]
    n = max(1, len(words))
    bigrams = list(zip(words, words[1:]))
    return dict(
        diversity=len(set(bigrams)) / max(1, len(bigrams)),
        valence=(sum(w in POS for w in words) - sum(w in NEG for w in words)) / n,
        hedging=sum(w in HEDGE for w in words) / n,
        self_ref=sum(w in ("i", "my", "me", "myself") for w in words) / n,
        fluency=logprob,
    )


def _clean_ppl(model, ids):
    return float(torch.exp(model(ids, labels=ids).loss))


def generate(model, tok, prompt, n=48, mode="strict", dose=0.08, body_id=0,
             run_seed=0, closed=False, base=0.03, k=0.03, H0=4.0):
    """mode: strict | noise | body.  closed=True -> dose driven by entropy (Exp B body)."""
    ids = tok(prompt, return_tensors="pt").input_ids
    n_prompt = ids.shape[1]
    logps, ents, cur = [], [], dose
    step_gen = torch.Generator().manual_seed(1234 + run_seed)
    for _ in range(n):
        seed = (body_id if mode == "body"
                else int(torch.randint(1 << 30, (1,), generator=step_gen)) if mode == "noise"
                else None)
        d = max(0.0, cur) if closed else dose
        h = install(model, "scale", d, seed=(seed if seed is not None else 0)) if (mode != "strict" and d > 0) else None
        logits = model(ids).logits[0, -1]
        probs = torch.softmax(logits, -1)
        H = float(-(probs * torch.log(probs + 1e-12)).sum())
        if h:
            remove(h)
        ents.append(H)
        nxt = logits.argmax().view(1)
        logps.append(float(torch.log(probs[nxt] + 1e-12)))
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
        if closed:
            cur = base + k * (H - H0)
    text = tok.decode(ids[0, n_prompt:])
    # max repeated-bigram run length (loop detector)
    bt = ids[0, n_prompt:].tolist()
    bg = list(zip(bt, bt[1:]))
    maxrun, run = 1, 1
    for i in range(1, len(bg)):
        run = run + 1 if bg[i] == bg[i - 1] else 1
        maxrun = max(maxrun, run)
    with torch.no_grad():
        ppl = _clean_ppl(model, ids)          # coherence under the CLEAN model
    return dict(text=text, logprob=float(np.mean(logps)), entropy=float(np.mean(ents)),
                maxrun=maxrun, distinct2=len(set(bg)) / max(1, len(bg)), ppl=ppl,
                **persona_features(text, float(np.mean(logps))))


if __name__ == "__main__":
    tok, model = load()
    PROMPTS = ["Once upon a time", "The scientist looked at the data and",
               "In the future, humanity will", "My favourite thing about winter is"]

    print("=" * 76)
    print("A. PERSONALITY — reproducibility & separability (body vs noise), dose=0.10")
    print("=" * 76)
    # reproducibility: run each mode twice, are the two runs identical?
    for mode in ["body", "noise"]:
        r1 = generate(model, tok, PROMPTS[0], mode=mode, dose=0.10, body_id=7, run_seed=1)
        r2 = generate(model, tok, PROMPTS[0], mode=mode, dose=0.10, body_id=7, run_seed=2)
        same = r1["text"] == r2["text"]
        print(f"  {mode:5s}: two runs identical? {same}   (body must be reproducible to be a persona)")
    # separability: can we tell body_id apart from persona features across prompts?
    print("\n  separability of 4 distinct bodies (feature spread across bodies vs within):")
    feats = {b: [] for b in [1, 2, 3, 4]}
    for b in feats:
        for pr in PROMPTS:
            feats[b].append(generate(model, tok, pr, mode="body", dose=0.12, body_id=b))
    import numpy as np
    for key in ["valence", "hedging", "self_ref", "diversity"]:
        per_body = [np.mean([f[key] for f in feats[b]]) for b in feats]
        print(f"    {key:10s}: bodies -> {[round(v,3) for v in per_body]}  spread={np.std(per_body):.4f}")

    print("\n  ORDERABLE DIAL — does dose monotonically shift a persona feature? (fixed body_id=3)")
    print("   dose   diversity  valence  hedging  fluency  ppl")
    for d in [0.0, 0.05, 0.1, 0.15, 0.2, 0.28]:
        rs = [generate(model, tok, pr, mode="body", dose=d, body_id=3) for pr in PROMPTS]
        agg = {k: np.mean([r[k] for r in rs]) for k in ["diversity", "valence", "hedging", "fluency", "ppl"]}
        print(f"   {d:4.2f}   {agg['diversity']:.3f}     {agg['valence']:+.3f}   {agg['hedging']:.3f}    {agg['fluency']:+.2f}   {agg['ppl']:.1f}")

    print("\n" + "=" * 76)
    print("B. TASK: LOOP-BREAKING — strict (loops) vs blind noise vs body (entropy-sensing)")
    print("=" * 76)
    print("  mode      maxrun  distinct2   ppl(coherence)   (lower maxrun + low ppl = win)")
    for mode, closed in [("strict", False), ("noise", False), ("body", True)]:
        rs = [generate(model, tok, pr, n=60, mode=mode, dose=0.06, body_id=5, closed=closed)
              for pr in PROMPTS]
        mr = np.mean([r["maxrun"] for r in rs]); d2 = np.mean([r["distinct2"] for r in rs])
        pp = np.mean([r["ppl"] for r in rs])
        tag = "sensing gain" if closed else ("bit-exact" if mode == "strict" else "blind noise")
        print(f"  {mode:6s} ({tag:12s}) {mr:5.1f}    {d2:.3f}      {pp:6.2f}")
    print("\n  sample (prompt 0):")
    for mode, closed in [("strict", False), ("body", True)]:
        r = generate(model, tok, PROMPTS[0], n=48, mode=mode, dose=0.06, body_id=5, closed=closed)
        print(f"    [{mode}] {r['text']!r}")
