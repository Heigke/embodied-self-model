#!/usr/bin/env python3
"""
Night-long sweep. Three phases, all checkpointed (resume-safe on reclaim):

  A  ABSTRACT cartography — (sign, K, d) regime map + Lyapunov, validated against
     the analytic boundary K_crit(d)=2 sin(pi/(2(2d+1))) and period T*=2(2d+1).
  B  CONCRETE closed loop with real body-biased SR arithmetic — regime {M,V,cut}
     x sign x delay x dose(kappa) x seed. Load-bearing (do-test), self-oscillation,
     Lyapunov, thermal range.
  C  READER — can a stateful variance-reader decode the body under regime V
     (mean preserved), and does state help? Sweep the V-dose gamma and the leak.

Run:  python -m numeric_rooting.run_night            # all phases
      python -m numeric_rooting.run_night A          # one phase
Resumes automatically from results/*.jsonl. Summaries -> results/summary_*.json.
"""
import sys, time, itertools
import numpy as np
from .system import abstract_loop, ConcreteLoop
from .reader import recoverability_curve
from .metrics import (periodicity, dominant_period, lyapunov_rosenstein,
                      participation_ratio, k_crit, predicted_period)
from . import checkpoint

RES = "numeric_rooting/results"
CLK = time.time

# neighbour profiles (rebuilt from params, since lambdas can't be checkpointed)
def const_neigh(level=0.10):
    return lambda t: level

def multisine_neigh(t):
    return 0.10 + 0.40 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 300)) \
                + 0.12 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 97))


# ---------------- phase A: abstract cartography ----------------
def run_A(p):
    x = abstract_loop(K=p["K"], d=p["d"], a=p["a"], sign=p["sign"],
                      slope=1.0, delta=1.0, noise=p["noise"],
                      T=3000, burn=1200, seed=p["seed"])
    diverged = not np.isfinite(x[-1]) or np.abs(x[-1]) > 40
    per = periodicity(x)
    return dict(_tag=f"A s{p['sign']} K{p['K']:.2f} d{p['d']}",
                std=round(float(np.std(x)), 5),
                periodicity=round(per, 4),
                period=dominant_period(x),
                lyap=round(lyapunov_rosenstein(x), 5) if not diverged else None,
                diverged=bool(diverged),
                k_crit=round(k_crit(p["d"]), 5),
                pred_period=predicted_period(p["d"]),
                osc=bool(per > 0.5 and np.std(x) > 1e-3 and not diverged))


def cells_A():
    Ks = np.round(np.geomspace(0.05, 3.0, 22), 4)
    ds = [0, 1, 2, 3, 4, 6, 8, 12]
    signs = [-1, +1]
    seeds = [0, 1, 2, 3]
    cells = []
    for sign, d, K, seed in itertools.product(signs, ds, Ks, seeds):
        cells.append(dict(phase="A", sign=sign, d=int(d), K=float(K),
                          a=1.0, noise=0.0, seed=seed))
    return cells


# ---------------- phase B: concrete closed loop ----------------
_CL_CACHE = {}
def _cl(seed):
    if seed not in _CL_CACHE:
        _CL_CACHE[seed] = ConcreteLoop(K=48, H=32, B=16, seed=seed)
    return _CL_CACHE[seed]


def run_B(p):
    cl = _cl(p["seed"])
    r = cl.run(regime=p["regime"], sign=p["sign"], kappa=p["kappa"],
               delay=p["delay"], m_base=7, gamma=1.0,
               neigh_fn=const_neigh(0.10), T_steps=2600, burn=700, seed=p["seed"] + 100)
    T = r["T"]; e = r["e"]; y = r["y"]
    # do-test load-bearing: does behaviour (y) track the body (e)?
    corr_ey = float(np.corrcoef(e, y)[0, 1]) if e.std() > 1e-9 and y.std() > 1e-9 else 0.0
    return dict(_tag=f"B {p['regime']} s{p['sign']} d{p['delay']} k{p['kappa']:.3f}",
                T_lo=round(float(T.min()), 2), T_hi=round(float(T.max()), 2),
                T_mean=round(float(T.mean()), 2), L_mean=round(float(r["L"].mean()), 3),
                y_mean=round(float(y.mean()), 5), y_std=round(float(y.std()), 5),
                corr_body_behaviour=round(corr_ey, 4),
                periodicity_T=round(periodicity(T), 4),
                period_T=dominant_period(T),
                lyap_T=round(lyapunov_rosenstein(T), 5),
                gain=round(float(r["gain"]), 3))


def cells_B():
    regimes = ["M", "V", "cut"]
    signs = [-1, +1]
    delays = [0, 1, 2, 3, 4, 6, 8]
    kappas = np.round(np.geomspace(0.002, 0.15, 11), 5)
    seeds = [0, 1, 2, 3, 4, 5]
    cells = []
    for regime, sign, delay, kappa, seed in itertools.product(regimes, signs, delays, kappas, seeds):
        # cut/V behaviour is sign-independent; keep only sign=-1 for them to avoid duplicates
        if regime in ("cut",) and sign == +1:
            continue
        cells.append(dict(phase="B", regime=regime, sign=int(sign),
                          delay=int(delay), kappa=float(kappa), seed=seed))
    return cells


# ---------------- phase C: interoceptive reader ----------------
def run_C(p):
    cl = _cl(p["seed"])
    r = cl.run(regime=p["regime"], sign=-1, kappa=0.05, delay=1, m_base=7,
               gamma=p["gamma"], neigh_fn=multisine_neigh,
               T_steps=3000, burn=600, seed=p["seed"] + 200)
    curve = recoverability_curve(r["logvar"], r["e"])
    lams = sorted(curve)
    best_lam = max(curve, key=curve.get)
    return dict(_tag=f"C {p['regime']} g{p['gamma']:.2f}",
                y_mean=round(float(r["y"].mean()), 5),
                y_std=round(float(r["y"].std()), 5),
                body_range=round(float(r["e"].max() - r["e"].min()), 3),
                hvar_lo=round(float(r["hvar"].min()), 5),
                hvar_hi=round(float(r["hvar"].max()), 5),
                pr=round(participation_ratio(r["logvar"]), 3),
                R2_instant=round(curve[1.0], 4),
                R2_best=round(max(curve.values()), 4),
                best_lam=best_lam,
                state_helps=bool(max(curve.values()) > curve[1.0] + 0.05),
                curve={str(k): round(v, 4) for k, v in curve.items()})


def cells_C():
    regimes = ["V", "M", "cut"]
    gammas = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0]
    seeds = [0, 1, 2, 3, 4]
    cells = []
    for regime, gamma, seed in itertools.product(regimes, gammas, seeds):
        cells.append(dict(phase="C", regime=regime, gamma=float(gamma), seed=seed))
    return cells


# ---------------- phase D: aggression ladder (behavioural effect vs coherence) ----------------
def run_D(p):
    """Precision-truncation ladder (plan 5c, rung A4). Forward-only characterisation:
       how far does behaviour move, and how much does the representation degrade (CKA),
       as effective precision falls? The gap between the two curves is the finding."""
    from .metrics import linear_cka
    cl = _cl(p["seed"])
    rng = np.random.default_rng(p["seed"] + 300)
    # pristine reference (full precision, no bias), averaged over draws
    ys0, H0 = [], []
    for _ in range(p["draws"]):
        y, h = cl.forward(7, 0.0, rng)
        ys0.append(y); H0.append(h)
    y0 = float(np.mean(ys0)); H0 = np.mean(H0, axis=0)
    # damaged at reduced mantissa m (optionally with a small mean-shift bias)
    ys, Hs = [], []
    for _ in range(p["draws"]):
        y, h = cl.forward(p["m"], p["b"], rng)
        ys.append(y); Hs.append(h)
    ym = float(np.mean(ys)); Hm = np.mean(Hs, axis=0)
    return dict(_tag=f"D m{p['m']} b{p['b']:.3f}",
                y_shift=round(ym - y0, 6),
                cka_vs_pristine=round(linear_cka(H0, Hm), 4),
                pr=round(participation_ratio(Hm), 3),
                hvar=round(float(Hm.var()), 5))


def cells_D():
    ms = [7, 6, 5, 4, 3, 2]
    bs = [0.0, 0.05, 0.15]          # V-like (b=0) and two M-like mean-shifts
    seeds = [0, 1, 2, 3, 4, 5]
    cells = []
    for m, b, seed in itertools.product(ms, bs, seeds):
        cells.append(dict(phase="D", m=int(m), b=float(b), seed=seed, draws=24))
    return cells


PHASES = {
    "A": (cells_A, run_A),
    "B": (cells_B, run_B),
    "C": (cells_C, run_C),
    "D": (cells_D, run_D),
}


def main(which=None):
    order = list(which) if which else ["A", "D", "C", "B"]   # cheap+informative first, big last
    for ph in order:
        make_cells, run_fn = PHASES[ph]
        cells = make_cells()
        print(f"\n==== PHASE {ph}: {len(cells)} cells ====", flush=True)
        checkpoint.run_grid(
            cells, run_fn,
            jsonl_path=f"{RES}/cells_{ph}.jsonl",
            summary_path=f"{RES}/summary_{ph}.json",
            summary_every=20, clock=CLK)
        print(f"==== PHASE {ph} complete ====", flush=True)
    print("\nALL PHASES COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
