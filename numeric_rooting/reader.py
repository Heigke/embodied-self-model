#!/usr/bin/env python3
"""
The interoceptive reader (Phase C): decode the body from the network's OWN
activation VARIANCE, with internal state — not from the body signal directly.

Rationale (Kornblith metrics agent + active-inference): under regime V the body
shifts the variance of the hidden layer, not its mean, so a single forward pass
is one sample and cannot reveal it. The reader must (a) track a running variance
feature (log-compressed, since variance changes are multiplicative) and (b) carry
a leaky recurrent state that integrates it over steps. Decode quality should RISE
as the integration window lengthens (leak lambda falls) — that curve is the proof
that persistent state is load-bearing, not decoration.
"""
import numpy as np


def leaky_integrate(Z, lam):
    """Z:(T,H) -> leaky-integrated (T,H). lam in (0,1]; 1 = no memory."""
    S = np.empty_like(Z)
    s = Z[0].copy()
    for t in range(len(Z)):
        s = (1 - lam) * s + lam * Z[t]
        S[t] = s
    return S


def decode_r2(Z, target, lam, train_frac=0.6, ridge=1e-2):
    """Leaky-integrate variance feature Z, ridge-regress -> target, return test R^2."""
    S = leaky_integrate(Z, lam)
    n = len(S); ntr = int(n * train_frac)
    Xtr, Xte = S[:ntr], S[ntr:]
    ytr, yte = target[:ntr], target[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
    return float(1 - ss_res / ss_tot)


def recoverability_curve(Z, target, lams=(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """Decode R^2 vs leak. Rising as lam falls => integration window (state) matters."""
    return {lam: decode_r2(Z, target, lam) for lam in lams}


if __name__ == "__main__":
    from .system import ConcreteLoop
    from .metrics import binning_mi
    cl = ConcreteLoop(seed=0)
    print("Reader on a regime-V run (mean uninformative; body lives in the variance)")
    r = cl.run(regime="V", sign=-1, kappa=0.05, delay=1, gamma=0.8, T_steps=2500, burn=600)
    Z = r["logvar"]; e = r["e"]
    print(f"  body range [{e.min():.1f},{e.max():.1f}]C  hidden-var range "
          f"[{r['hvar'].min():.3f},{r['hvar'].max():.3f}]")
    curve = recoverability_curve(Z, e)
    print("  leak lambda -> decode R^2 (window ~ 1/lambda):")
    for lam, r2 in curve.items():
        print(f"    lambda={lam:5.2f}  window~{1/lam:5.0f}  R^2={r2:+.3f}")
    best = max(curve.values()); inst = curve[1.0]
    print(f"  instantaneous (no state) R^2={inst:+.3f}  vs best-integrated R^2={best:+.3f}  "
          f"-> {'STATE HELPS' if best > inst + 0.05 else 'state does not help here'}")
