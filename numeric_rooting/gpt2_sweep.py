#!/usr/bin/env python3
"""
GPT-2 aggression-ladder sweep — behaviour vs coherence on a real transformer.

Per cell we measure the plan's two curves plus the unbiasedness test:
  kl_single    KL(pristine || perturbed), single rounding seed   -> per-pass behaviour move
  kl_seedmean  KL(pristine || mean over N seeds)                 -> is it a MEAN effect (M/scale)
                                                                    or noise that averages away (V)?
  logit_shift  mean |logit change|
  ppl          perplexity on held text (coherence); ppl_ratio vs pristine; broke = ppl>2x

Sub-sweeps:
  ladder  all layers, 4 regimes (M/V/scale/signflip) x dose ladder to destruction
  depth   single-layer, regime x layer x a few doses -> where the body has leverage
  fine    fine dose ladder for V and scale -> locate the coherence wall precisely

Checkpointed + self-persisting (survives container restarts). Resumes from results/.
Run:  python -m numeric_rooting.gpt2_sweep
"""
import sys, time, itertools, subprocess
import numpy as np
import torch
from .gpt2_l0 import load, install, remove, next_token_dist, perplexity, kl
from . import checkpoint

RES = "numeric_rooting/results"
BRANCH = "claude/numeric-rooting-mac-5qy5xm"
_REF = {}


def _ref():
    if not _REF:
        tok, model = load()
        P0, L0 = next_token_dist(model, tok)
        _REF["tok"], _REF["model"] = tok, model
        _REF["P0"], _REF["L0"] = P0, L0
        _REF["ppl0"] = perplexity(model, tok)
    return _REF


def run_cell(p):
    r = _ref(); model, tok = r["model"], r["tok"]
    layers = None if p.get("layer", -1) < 0 else [p["layer"]]
    nseeds = p.get("seeds", 12)
    Ps, Ls, ppls = [], [], []
    for s in range(nseeds):
        h = install(model, p["regime"], p["dose"], layers=layers, seed=s)
        P, L = next_token_dist(model, tok)
        ppls.append(perplexity(model, tok))
        remove(h)
        Ps.append(P); Ls.append(L)
    Pmean = torch.stack(Ps).mean(0)
    ppl = float(np.mean(ppls))
    return dict(_tag=f"{p['regime']} d{p['dose']} L{p.get('layer',-1)}",
                kl_single=round(kl(r["P0"], Ps[0]), 5),
                kl_seedmean=round(kl(r["P0"], Pmean), 5),
                logit_shift=round(float((Ls[0] - r["L0"]).abs().mean()), 4),
                ppl=round(ppl, 3), ppl0=round(r["ppl0"], 3),
                ppl_ratio=round(ppl / r["ppl0"], 4),
                broke=bool(ppl > 2 * r["ppl0"]))


def cells_ladder():
    ladders = {
        "M": [0.05, 0.1, 0.2, 0.4, 0.8, 1.5],
        "V": [1, 2, 3, 4, 5],
        "scale": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
        "signflip": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5],
    }
    return [dict(phase="ladder", regime=r, dose=float(d), layer=-1, seeds=12)
            for r, ds in ladders.items() for d in ds]


def cells_depth():
    combos = {"V": [3], "scale": [0.05], "signflip": [0.03]}
    return [dict(phase="depth", regime=r, dose=float(d), layer=int(L), seeds=8)
            for r, ds in combos.items() for d in ds for L in range(12)]


def cells_fine():
    fine = {"V": list(np.round(np.linspace(0, 5.5, 12), 3)),
            "scale": list(np.round(np.geomspace(0.005, 1.2, 12), 4))}
    return [dict(phase="fine", regime=r, dose=float(d), layer=-1, seeds=12)
            for r, ds in fine.items() for d in ds]


def git_persist(msg):
    try:
        subprocess.run("git add -f numeric_rooting/results/cells_gpt2_*.jsonl "
                       "numeric_rooting/results/summary_gpt2_*.json", shell=True, check=False,
                       timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-q", "-m", msg], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "push", "origin", BRANCH], check=False, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


PHASES = {"ladder": cells_ladder, "depth": cells_depth, "fine": cells_fine}


def main(which=None):
    order = list(which) if which else ["ladder", "depth", "fine"]
    for ph in order:
        cells = PHASES[ph]()
        print(f"\n==== GPT-2 PHASE {ph}: {len(cells)} cells ====", flush=True)
        checkpoint.run_grid(
            cells, run_cell,
            jsonl_path=f"{RES}/cells_gpt2_{ph}.jsonl",
            summary_path=f"{RES}/summary_gpt2_{ph}.json",
            summary_every=15, clock=time.time,
            on_summary=lambda recs, ph=ph: git_persist(f"gpt2 sweep auto-checkpoint: {ph}, {len(recs)} cells"))
        git_persist(f"gpt2 sweep: phase {ph} complete")
        print(f"==== GPT-2 PHASE {ph} complete ====", flush=True)
    print("\nGPT-2 SWEEP COMPLETE", flush=True)
    git_persist("gpt2 sweep: COMPLETE")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
