#!/usr/bin/env python3
"""
L0 on a real transformer (GPT-2 small, 124M) — the experiment FINDINGS.md points to.

Per the plan (5b, level L0): keep the matmul in fp32, but apply BODY-BIASED STOCHASTIC
ROUNDING to each matmul's OUTPUT via a PyTorch forward hook. This is the reachable
rounding site (accumulator -> storage), ~10 lines, no custom kernel.

Two regimes, both inside the rounding, both a `dose`:
  M : round up with p = theta + b   (mean-shift b -> changes the function)
  V : round to m_eff mantissa bits   (precision truncation; unbiased in the mean)

We measure the plan's two curves on the SAME axis:
  behavioural effect  = KL( pristine next-token dist || perturbed )   [does behaviour move?]
  coherence           = perplexity on held text                       [does it stay sensible?]
plus mean |logit shift| to separate M (mean moves) from V (variance only, mean ~preserved),
and a depth profile (apply only at layer L) to find where the body has most leverage.
"""
import torch, torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_TOK = None
_MODEL = None


def load():
    global _TOK, _MODEL
    if _MODEL is None:
        _TOK = GPT2TokenizerFast.from_pretrained("gpt2")
        _MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        torch.set_grad_enabled(False)
    return _TOK, _MODEL


def sr_round_torch(x, m, b, gen):
    """Body-biased stochastic rounding of x to m explicit mantissa bits.
       b: mean-shift (regime M). b=0 -> unbiased (regime V uses small m instead)."""
    mant, expo = torch.frexp(x)                     # x = mant * 2**expo, |mant| in [0.5,1)
    s = torch.ldexp(mant, torch.tensor(m + 1))
    f = torch.floor(s)
    theta = s - f
    p = torch.clamp(theta + b, 0.0, 1.0)
    u = torch.rand(x.shape, generator=gen)
    up = ((u < p) & (theta > 0)).to(x.dtype)
    return torch.ldexp(f + up, expo - (m + 1))


def target_modules(model, layers=None):
    """The Conv1D matmuls in each block: attention c_attn/c_proj, MLP c_fc/c_proj."""
    mods = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "Conv1D":
            li = _layer_index(name)
            if layers is None or li in layers:
                mods.append((name, mod))
    return mods


def _layer_index(name):
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def install(model, regime, dose, layers=None, seed=0):
    """Attach a body-modulated arithmetic intervention to matmul outputs.
       Aggression ladder (plan 5c), gentle -> destructive:
         M       mean-shift stochastic rounding  (dose = bias b; mantissa 7)
         V       precision truncation            (dose = mantissa bits dropped)
         scale   multiplicative gain jitter       (dose = std of (1+dose*noise))
         signflip sign corruption                 (dose = fraction of elements flipped)
       M/V live inside the rounding; scale/signflip are the far, destructive rungs."""
    gen = torch.Generator().manual_seed(seed)
    handles = []
    m = int(max(2, round(7 - dose))) if regime == "V" else 7
    b = float(dose) if regime == "M" else 0.0

    def hook(mod, inp, out):
        if regime in ("M", "V"):
            return sr_round_torch(out, m, b, gen)
        if regime == "scale":
            return out * (1.0 + dose * torch.randn(out.shape, generator=gen))
        if regime == "signflip":
            flip = (torch.rand(out.shape, generator=gen) < dose)
            return torch.where(flip, -out, out)
        return out

    for _, mod in target_modules(model, layers):
        handles.append(mod.register_forward_hook(hook))
    return handles


def remove(handles):
    for h in handles:
        h.remove()


# ---------------- measurements ----------------
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "In the morning I like to drink a cup of",
    "The opposite of hot is",
    "Two plus two equals",
    "She opened the door and saw a",
]
HELD_TEXT = (
    "The sun rose over the quiet village as farmers walked to their fields. "
    "Children laughed on their way to school, carrying books and small lunches. "
    "By noon the market was full of people buying bread, fruit, and fresh fish."
)


def next_token_dist(model, tok):
    """Softmax over the last-position logits for each prompt -> (P, mean_logits)."""
    dists, logits_all = [], []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        lg = model(ids).logits[0, -1]
        logits_all.append(lg)
        dists.append(torch.softmax(lg, -1))
    return torch.stack(dists), torch.stack(logits_all)


def perplexity(model, tok, text=HELD_TEXT):
    ids = tok(text, return_tensors="pt").input_ids
    loss = model(ids, labels=ids).loss
    return float(torch.exp(loss))


def kl(p_ref, p):
    return float((p_ref * (torch.log(p_ref + 1e-12) - torch.log(p + 1e-12))).sum(-1).mean())


if __name__ == "__main__":
    tok, model = load()
    P0, L0 = next_token_dist(model, tok)
    ppl0 = perplexity(model, tok)
    print(f"pristine: perplexity={ppl0:.2f}")
    print("\nregime M (mean-shift b) — all layers:")
    print("   dose b   KL(behaviour)   |logit shift|   perplexity(coherence)")
    for b in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        h = install(model, "M", b); P, L = next_token_dist(model, tok); ppl = perplexity(model, tok); remove(h)
        print(f"   {b:5.2f}    {kl(P0, P):.4f}         {float((L-L0).abs().mean()):.4f}          {ppl:.2f}")
    print("\nregime V (bits dropped) — all layers:")
    print("   bits↓   KL(behaviour)   |logit shift|   perplexity(coherence)")
    for d in [0, 1, 2, 3, 4, 5]:
        h = install(model, "V", d); P, L = next_token_dist(model, tok); ppl = perplexity(model, tok); remove(h)
        print(f"   {d:3d}     {kl(P0, P):.4f}         {float((L-L0).abs().mean()):.4f}          {ppl:.2f}")
