#!/usr/bin/env python3
"""
Deep rooting — the body INSIDE the multiply-accumulate, not on its output.

L0 (gpt2_l0.py) rounds the finished matmul output. This goes one level deeper:
it replaces GPT-2's Conv1D matmul with an explicit SPLIT-K accumulation and roots
the body into the accumulation itself — the only information-destroying step the
plan identifies (product exact, add exact, ROUND is the choice). Three sites:

  round  body-biased stochastic rounding of the RUNNING ACCUMULATOR after each
         K-block  -> the per-FMA rounding, at block granularity
  gain   body-modulated multiplicative gain on each partial sum before it is
         accumulated  -> the regime GPT-2 actually reads (FINDINGS: scale)
  order  body-dependent ORDER of block summation, no rounding change  -> Site B,
         "physics already leaking through" (fp add is non-associative)

Reachable on CPU because the token count is small; the K-loop is over ~8 blocks,
not thousands, so each split-k matmul is a handful of exact sub-matmuls plus a
body op. Deeper than L0, same measurement harness.
"""
import torch
from .gpt2_l0 import sr_round_torch, target_modules

_ORIG = {}   # module -> original forward


def _blocks(K, nblocks):
    step = (K + nblocks - 1) // nblocks
    return [slice(i, min(i + step, K)) for i in range(0, K, step)]


def _make_forward(mod, regime, dose, nblocks, gen):
    W = mod.weight                      # (nx, nf)
    bias = mod.bias                     # (nf,)
    nx, nf = W.shape
    blocks = _blocks(nx, nblocks)
    m = int(max(2, round(7 - dose))) if regime == "roundV" else 7
    b = float(dose) if regime == "round" else 0.0

    def forward(x):
        shape = x.size()[:-1] + (nf,)
        x2 = x.reshape(-1, nx)
        acc = torch.zeros(x2.shape[0], nf, dtype=x2.dtype)
        order = list(range(len(blocks)))
        if regime == "order":
            perm = torch.randperm(len(blocks), generator=gen)
            order = perm.tolist()
        for bi in order:
            sl = blocks[bi]
            partial = x2[:, sl] @ W[sl, :]            # exact fp32 partial product
            if regime == "gain":
                partial = partial * (1.0 + dose * torch.randn(partial.shape, generator=gen))
            acc = acc + partial
            if regime in ("round", "roundV"):
                acc = sr_round_torch(acc, m, b, gen)   # round the RUNNING accumulator
        return (acc + bias).reshape(shape)

    return forward


def install_deep(model, regime, dose, nblocks=8, layers=None, seed=0):
    """Monkeypatch target Conv1D matmuls with a body-rooted split-k accumulation."""
    gen = torch.Generator().manual_seed(seed)
    for _, mod in target_modules(model, layers):
        if mod not in _ORIG:
            _ORIG[mod] = mod.forward
        mod.forward = _make_forward(mod, regime, dose, nblocks, gen)


def remove_deep(model):
    for mod, fwd in list(_ORIG.items()):
        mod.forward = fwd
    _ORIG.clear()


if __name__ == "__main__":
    from .gpt2_l0 import load, next_token_dist, perplexity, kl
    tok, model = load()
    P0, L0 = next_token_dist(model, tok); ppl0 = perplexity(model, tok)
    print(f"pristine perplexity={ppl0:.2f}")

    # correctness: split-k with no body op must reproduce the model exactly
    install_deep(model, "order", 0.0, nblocks=4); P, _ = next_token_dist(model, tok); remove_deep(model)
    # 'order' with a fixed gen still permutes; use a no-op check instead:
    install_deep(model, "round", 0.0, nblocks=1); P1, _ = next_token_dist(model, tok)
    ppl1 = perplexity(model, tok); remove_deep(model)
    print(f"sanity (round b=0, 1 block): KL={kl(P0,P1):.5f} ppl={ppl1:.2f}  (expect ~0 / ~{ppl0:.1f})")

    print("\nDEEP rooting inside the accumulation (8 blocks), all layers:")
    print("  site    dose    KL(behaviour)   perplexity")
    for regime, dose in [("round",0.1),("round",0.4),("gain",0.02),("gain",0.05),
                         ("gain",0.1),("order",0.0)]:
        install_deep(model, regime, dose, nblocks=8); P, _ = next_token_dist(model, tok)
        ppl = perplexity(model, tok); remove_deep(model)
        print(f"  {regime:6s} {dose:5.2f}    {kl(P0,P):.5f}        {ppl:.2f}")
