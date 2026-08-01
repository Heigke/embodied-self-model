#!/usr/bin/env python3
"""
Measurement suite. numpy-only, night-run cheap.

Dynamical:  periodicity, largest Lyapunov exponent (Rosenstein), participation ratio.
Representation:  linear CKA (Kornblith 2019), binning mutual information.
Control theory:  analytic delay stability boundary K_crit(d) (integrator case).
"""
import numpy as np


# ---------- dynamical ----------
def periodicity(x, lo=3, hi=80):
    """Peak |autocorrelation| at lag>=lo. ~1 for a clean limit cycle, ~0 for noise."""
    x = np.asarray(x, float); x = x - x.mean()
    if x.std() < 1e-12 or len(x) < hi + 2:
        return 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    return float(np.max(np.abs(ac[lo:hi])))


def dominant_period(x, lo=2, hi=200):
    """Lag of the peak autocorrelation (samples). 0 if none."""
    x = np.asarray(x, float); x = x - x.mean()
    if x.std() < 1e-12 or len(x) < hi + 2:
        return 0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    return int(lo + np.argmax(ac[lo:hi]))


def lyapunov_rosenstein(x, m=4, tau=1, theiler=None, k_max=None, fit=(1, 12)):
    """Largest Lyapunov exponent from a scalar series (Rosenstein 1993).
       >0 chaotic, ~0 marginal/limit-cycle, <0 fixed point. Units: per sample."""
    x = np.asarray(x, float)
    N = len(x) - (m - 1) * tau
    if N < 50:
        return float("nan")
    Y = np.stack([x[i:i + N] for i in range(0, m * tau, tau)], axis=1)  # (N,m)
    theiler = theiler or (tau * m)
    k_max = k_max or min(40, N // 3)
    # nearest neighbour outside the Theiler window, per point
    div = np.zeros(k_max)
    cnt = np.zeros(k_max)
    for i in range(N):
        d = np.sqrt(((Y - Y[i]) ** 2).sum(1))
        d[max(0, i - theiler):i + theiler + 1] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        kk = min(k_max, N - max(i, j))
        sep = np.sqrt(((Y[i:i + kk] - Y[j:j + kk]) ** 2).sum(1))
        sep = np.where(sep > 0, sep, np.nan)
        ln = np.log(sep)
        good = np.isfinite(ln)
        div[:kk][good] += np.nan_to_num(ln[good])
        cnt[:kk][good] += 1
    y = np.divide(div, cnt, out=np.full(k_max, np.nan), where=cnt > 0)
    a, b = fit
    b = min(b, k_max)
    ks = np.arange(a, b)
    yy = y[a:b]
    ok = np.isfinite(yy)
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(ks[ok], yy[ok], 1)[0]
    return float(slope)


def participation_ratio(H):
    """Effective dimensionality of activations H (n_samples, d). PR in [1,d]."""
    H = np.asarray(H, float)
    if H.ndim == 1 or H.shape[0] < 2:
        return 1.0
    C = np.cov(H, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    s2 = (ev ** 2).sum()
    return float(ev.sum() ** 2 / s2) if s2 > 0 else 1.0


# ---------- representation ----------
def linear_cka(X, Y):
    """Linear CKA between representations X (n,p1), Y (n,p2). In [0,1]."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro") ** 2
    yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    den = np.sqrt(xx * yy)
    return float(xy / den) if den > 0 else 0.0


def binning_mi(x, y, bins=16):
    """Mutual information (nats) between scalar series x, y via 2-D histogram."""
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    c = np.histogram2d(x, y, bins)[0]
    p = c / c.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / (px @ py)[m])))


# ---------- control theory ----------
def k_crit(d):
    """Analytic loop-gain stability boundary for the delayed integrator map
       x_{t+1} = x_t - K*sat(x_{t-d}):  stable iff K < 2*sin(pi/(2*(2d+1)))."""
    return 2.0 * np.sin(np.pi / (2.0 * (2 * d + 1)))


def predicted_period(d):
    """Emergent oscillation period at onset (samples): T* = 2*(2d+1)."""
    return 2 * (2 * d + 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    sine = np.sin(2 * np.pi * t / 20) + 0.01 * rng.standard_normal(2000)
    noise = rng.standard_normal(2000)
    logistic = np.zeros(2000); logistic[0] = 0.4
    for i in range(1999):
        logistic[i + 1] = 3.99 * logistic[i] * (1 - logistic[i])   # chaotic
    print(f"periodicity  sine={periodicity(sine):.2f}  noise={periodicity(noise):.2f}")
    print(f"dominant_period sine={dominant_period(sine)} (expect 20)")
    print(f"lyapunov  sine={lyapunov_rosenstein(sine):+.3f}  noise={lyapunov_rosenstein(noise):+.3f}  "
          f"logistic={lyapunov_rosenstein(logistic):+.3f} (expect >0 ~0.69)")
    X = rng.standard_normal((500, 8)); print(f"CKA(X,X)={linear_cka(X, X):.2f} (expect 1)  "
          f"CKA(X,rand)={linear_cka(X, rng.standard_normal((500, 8))):.2f}")
    print(f"PR(iso 8-d)={participation_ratio(rng.standard_normal((1000,8))):.1f} (expect ~8)")
    print(f"k_crit(d=0)={k_crit(0):.3f} k_crit(d=1)={k_crit(1):.3f} k_crit(d=8)={k_crit(8):.3f}")
