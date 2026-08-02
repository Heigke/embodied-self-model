#!/usr/bin/env python3
"""
Can a signal be LIFTED out of what looks like noise? — the crux of "not noise".

Regime V is unbiased: per forward pass it looks like noise, and its mean effect
averages away. But if the body is a COHERENT modulator — the same signal b(t)
scaling many units' fluctuation at once — then advanced analysis can recover b(t)
even though any single unit sees only noise, while matched INCOHERENT noise of the
same magnitude yields nothing. That gap is the difference between "brus in" and a
signal a downstream circuit can read (interoception in the correct sense).

Setup: inject mean-zero fluctuation at GPT-2 layer L_inj whose STD is modulated by
b(t) (unbiased — mean preserved), read the population at a downstream layer L_read
over T passes, and try to recover b(t) by:
  1. population energy  E[t] = mean_units (a_t - a_clean)^2   (cross-unit averaging)
  2. temporal EMA smoothing                                    (integration over time)
Conditions:
  body        b(t) scales ALL units together (coherent, low-rank, slow)
  incoherent  same magnitude, but b time-shuffled independently per unit (no shared signal)
  white       constant std, no b(t)                            (pure noise, nothing to recover)
Analyses: R^2(recovered, true b); R^2 vs number of units averaged (the sqrt(N) signature
of a coherent signal in noise); spectral peak; mutual information.
"""
import torch
torch.set_num_threads(2)
import numpy as np
from .gpt2_l0 import load
from .metrics import binning_mi

L_INJ, L_READ = 4, 8
PROMPT = "The weather today is"


def _b(T):
    t = np.arange(T)
    b = 0.5 + 0.4 * np.sin(2 * np.pi * t / 60) + 0.2 * np.sin(2 * np.pi * t / 23)
    return (b - b.min()) / (b.max() - b.min())          # in [0,1]


def run(model, tok, T=320, amp=1.2, s0=0.4, cond="body", seed=0):
    rng = np.random.default_rng(seed)
    b = _b(T)
    H = model.config.n_embd
    inj = model.transformer.h[L_INJ].mlp.c_proj
    read = model.transformer.h[L_READ].mlp.c_proj
    state = {"scale": np.zeros(H)}
    cap = {}
    # per-unit time-shuffled b for the incoherent control
    shuffle = np.stack([rng.permutation(T) for _ in range(H)], 1)   # (T,H) indices

    def inj_hook(mod, i, out):
        sc = torch.tensor(state["scale"], dtype=out.dtype)
        return out + sc * torch.randn(out.shape, generator=state["gen"])

    def read_hook(mod, i, out):
        cap["a"] = out[0, -1, :].detach().clone().numpy()

    ids = tok(PROMPT, return_tensors="pt").input_ids
    state["gen"] = torch.Generator().manual_seed(1000 + seed)
    # clean reference (no injection)
    hr = read.register_forward_hook(read_hook)
    state["scale"] = np.zeros(H); model(ids); a_clean = cap["a"].copy()
    hi = inj.register_forward_hook(inj_hook)
    A = np.zeros((T, H))
    for t in range(T):
        if cond == "white":
            sc = s0 * np.ones(H)
        elif cond == "incoherent":
            sc = s0 * (1 + amp * b[shuffle[t]])           # each unit sees a different time-scramble of b
        else:  # body: coherent shared modulator
            sc = s0 * (1 + amp * b[t]) * np.ones(H)
        state["scale"] = sc
        model(ids)
        A[t] = cap["a"]
    hi.remove(); hr.remove()
    R = A - a_clean                                        # residual fluctuation
    return b, R


def ema(x, lam=0.15):
    y = np.empty_like(x); s = x[0]
    for i, v in enumerate(x):
        s = (1 - lam) * s + lam * v; y[i] = s
    return y


def recover(R, n_units=None, rng=None):
    """Population energy across units, temporally smoothed -> estimate of b(t)."""
    H = R.shape[1]
    if n_units and n_units < H:
        idx = (rng or np.random.default_rng(0)).choice(H, n_units, replace=False)
        R = R[:, idx]
    E = (R ** 2).mean(1)                                   # cross-unit averaging
    return ema(E, 0.15)


def r2(true, est):
    b = (true - true.mean()); e = (est - est.mean())
    if b.std() < 1e-9 or e.std() < 1e-9:
        return 0.0
    c = np.corrcoef(b, e)[0, 1]
    return float(c ** 2)


if __name__ == "__main__":
    tok, model = load()
    print(f"inject fluctuation at layer {L_INJ}, read population at layer {L_READ}, T=320 passes\n")
    print("Recover b(t) from the population — body (coherent) vs matched controls:")
    print("  condition     R^2(recovered,b)   MI(recovered;b)   note")
    res = {}
    for cond in ["body", "incoherent", "white"]:
        b, R = run(model, tok, cond=cond)
        est = recover(R)
        res[cond] = (b, R)
        note = {"body": "coherent modulator", "incoherent": "same magnitude, no shared signal",
                "white": "constant std, no b(t)"}[cond]
        print(f"  {cond:11s}   {r2(b, est):.3f}              {binning_mi(b, est):.3f}            {note}")

    print("\nsqrt(N) signature — R^2 vs number of units averaged (coherent signal grows with N):")
    print("  N_units:   ", "  ".join(f"{n:4d}" for n in [1, 4, 16, 64, 256, 768]))
    for cond in ["body", "incoherent"]:
        b, R = res[cond]
        rng = np.random.default_rng(1)
        row = [r2(b, recover(R, n, rng)) for n in [1, 4, 16, 64, 256, 768]]
        print(f"  {cond:11s}" + "  ".join(f"{v:.2f}" for v in row).rjust(0))

    print("\nspectral check — is b(t)'s slow rhythm present in the recovered signal?")
    for cond in ["body", "incoherent", "white"]:
        b, R = res[cond]
        est = recover(R)
        fb = np.abs(np.fft.rfft(b - b.mean())); fe = np.abs(np.fft.rfft(est - est.mean()))
        peak = int(np.argmax(fb[1:]) + 1)
        frac = fe[peak] / (fe[1:].sum() + 1e-9)
        print(f"  {cond:11s}: energy at b's peak frequency = {frac:.3f} of recovered spectrum")

    print("\n=> if body >> incoherent/white, a coherent body signal is LIFTABLE from noise-like")
    print("   fluctuation by population averaging + temporal integration — a downstream unit")
    print("   reading its own population variance can act on the body it could never see per-unit.")
