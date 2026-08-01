#!/usr/bin/env python3
"""
Body-biased stochastic rounding — the arithmetic site where the body enters.

From the SR error analysis (Croci/Higham/Mary 2022; Connolly/Higham/Mary 2021):
round x up with probability p = clip(theta + b, 0, 1), theta = exact fractional
position on the target grid.
  b = 0  -> E[SR(x)] = x                    REGIME V (unbiased; safe; unread in mean)
  b != 0 -> E[SR(x)] = x + b*ulp(x)         REGIME M (mean shift -> changes the function)

Over a length-K accumulation the mean shift grows ~ b*K*ulp while the zero-mean
noise floor grows ~ sqrt(K)*ulp, so detectability (shift/std) ~ b*sqrt(K).

Regime V from a PHYSICAL signal must use the probability-integral transform
(rank -> exact Uniform(0,1)) so temporal correlation in the signal cannot leak
into the mean. Correlation between the signal and the data is exactly what a V
run must avoid; an M run injects a *mean* deliberately instead.
"""
import numpy as np


def sr_round(x, m, u, b=0.0, perturb_exact=False):
    """Round float64 array x to a format with m explicit mantissa bits.
       u: uniform(0,1) draws, x-shaped (PRNG, or rank-transformed physical signal).
       b: mean-shift knob (scalar or x-shaped). b=0 -> V, b!=0 -> M."""
    x = np.asarray(x, dtype=np.float64)
    mant, expo = np.frexp(x)                 # x = mant * 2**expo, |mant| in [0.5,1)
    s = np.ldexp(mant, m + 1)                # target grid points are the integers
    f = np.floor(s)
    theta = s - f                            # exact fractional position in [0,1)
    p = np.clip(theta + b, 0.0, 1.0)
    up = (u < p)
    if not perturb_exact:
        up = up & (theta > 0)                # identity on already-representable values
    return np.ldexp(f + up, expo - (m + 1))


def rank_uniform(z):
    """Probability-integral transform: physical samples -> exact Uniform(0,1)."""
    z = np.asarray(z).ravel()
    return (np.argsort(np.argsort(z)) + 0.5) / z.size


def sr_matvec(W, x, m, b, rng, u=None):
    """y = W @ x with SR after each accumulation step (row-wise, vectorised over rows).
       W: (out,K)  x: (K,)  ->  y: (out,).  b: mean-shift knob (regime selector)."""
    out, K = W.shape
    acc = np.zeros(out)
    prod = W * x[None, :]                     # (out,K) exact products
    for k in range(K):
        uu = rng.random(out) if u is None else u[:, k]
        acc = sr_round(acc + prod[:, k], m, uu, b)
    return acc


if __name__ == "__main__":
    # sanity checks from the SR analysis
    rng = np.random.default_rng(0)
    K, m = 512, 7                             # bf16 mantissa
    u_unit = 2.0 ** -(m + 1)
    A = rng.standard_normal((4000, K)) * 0.1
    B = rng.standard_normal((4000, K)) * 0.1
    exact = (A * B).sum(1)

    def dot(b):
        acc = np.zeros(A.shape[0])
        for k in range(K):
            acc = sr_round(acc + A[:, k] * B[:, k], m, rng.random(A.shape[0]), b)
        return acc

    yV = dot(0.0)                             # regime V
    yM = dot(0.02)                            # regime M
    print(f"K={K} m={m} unit_roundoff u={u_unit:.2e}")
    print(f"V: mean error = {(yV-exact).mean():+.2e}  (expect ~0)   "
          f"std error = {(yV-exact).std():.2e}  (~sqrt(K)*u*|y| = {np.sqrt(K)*u_unit*np.abs(exact).mean():.2e})")
    print(f"M(b=0.02): mean shift = {(yM-exact).mean():+.2e}  "
          f"(expect ~2*b*u*K*|y| = {2*0.02*u_unit*K*np.abs(exact).mean():+.2e})")
    print("OK" if abs((yV-exact).mean()) < (yM-exact).mean() else "CHECK")
