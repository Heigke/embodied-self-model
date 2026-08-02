#!/usr/bin/env python3
"""
Deep-rooting sweep — does rooting the body INSIDE the accumulation change the
verdict, and how does depth-of-rooting (split-k block count) matter?

Questions:
  Q1 round : sweep nblocks -> does rounding the running accumulator more often
             (finer split-k) increase load-bearing? Compare to L0 output rounding.
  Q2 gain  : sweep nblocks -> is the load-bearing gain best at the output (1 block)
             or distributed through the accumulation (many blocks)?
  Q3 order : sweep nblocks, and order+low-precision -> does accumulation order
             (Site B, non-associativity) ever move behaviour?

Per cell: kl_single, kl_seedmean (unbiasedness), ppl (coherence). Checkpointed,
self-persisting. Run: python -m numeric_rooting.gpt2_deep_sweep
"""
import sys, time, subprocess
import numpy as np
import torch
from .gpt2_l0 import load, next_token_dist, perplexity, kl
from .matmul_deep import install_deep, remove_deep
from . import checkpoint

RES = "numeric_rooting/results"
BRANCH = "claude/numeric-rooting-mac-5qy5xm"
_REF = {}


def _ref():
    if not _REF:
        tok, model = load()
        P0, L0 = next_token_dist(model, tok)
        _REF.update(tok=tok, model=model, P0=P0, L0=L0, ppl0=perplexity(model, tok))
    return _REF


def run_cell(p):
    r = _ref(); model, tok = r["model"], r["tok"]
    nseeds = p.get("seeds", 6)
    Ps, ppls = [], []
    for s in range(nseeds):
        install_deep(model, p["regime"], p["dose"], nblocks=p["nblocks"], seed=s)
        P, _ = next_token_dist(model, tok)
        ppls.append(perplexity(model, tok))
        remove_deep(model)
        Ps.append(P)
    Pmean = torch.stack(Ps).mean(0)
    ppl = float(np.mean(ppls))
    return dict(_tag=f"{p['regime']} d{p['dose']} nb{p['nblocks']}",
                kl_single=round(kl(r["P0"], Ps[0]), 5),
                kl_seedmean=round(kl(r["P0"], Pmean), 5),
                ppl=round(ppl, 3), ppl_ratio=round(ppl / r["ppl0"], 4),
                broke=bool(ppl > 2 * r["ppl0"]))


def cells():
    out = []
    for nb in [1, 2, 4, 8, 16, 32]:                       # Q1: rounding granularity
        for d in [0.2, 0.5]:
            out.append(dict(phase="deep", regime="round", dose=float(d), nblocks=nb, seeds=6))
    for nb in [1, 2, 4, 8, 16, 32]:                       # Q2: where the gain belongs
        for d in [0.03, 0.07]:
            out.append(dict(phase="deep", regime="gain", dose=float(d), nblocks=nb, seeds=6))
    for nb in [4, 16, 64]:                                # Q3a: pure order (fp32)
        out.append(dict(phase="deep", regime="order", dose=0.0, nblocks=nb, seeds=6))
    for nb in [4, 16, 64]:                                # Q3b: order under low precision
        out.append(dict(phase="deep", regime="roundV", dose=4.0, nblocks=nb, seeds=6))
    return out


def git_persist(msg):
    try:
        subprocess.run("git add -f numeric_rooting/results/cells_gpt2deep.jsonl "
                       "numeric_rooting/results/summary_gpt2deep.json", shell=True, check=False,
                       timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-q", "-m", msg], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "push", "origin", BRANCH], check=False, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    cs = cells()
    print(f"==== GPT-2 DEEP sweep: {len(cs)} cells ====", flush=True)
    checkpoint.run_grid(cs, run_cell,
                        jsonl_path=f"{RES}/cells_gpt2deep.jsonl",
                        summary_path=f"{RES}/summary_gpt2deep.json",
                        summary_every=10, clock=time.time,
                        on_summary=lambda recs: git_persist(f"gpt2 deep sweep: {len(recs)} cells"))
    git_persist("gpt2 deep sweep: COMPLETE")
    print("\nGPT-2 DEEP SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
