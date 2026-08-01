#!/usr/bin/env python3
"""
NUMERIC ROOTING as a closed loop — the simplest reglersystem that has the
computation depend on the hardware that runs it, and be changed by it.
=====================================================================

The plan (NUMERIC ROOTING, 2026-07-31) roots the body at the rounding decision
inside the multiply-accumulate: the physical hardware state biases the rounding,
which changes what the operation COMPUTES. That is a feedback interconnection,
not a wire you read. Two dynamical systems close on each other:

    HARDWARE PLANT  H:  input = compute-load u   state = h (contention/heat)
    COMPUTATION     M:  input = body-bias  b     output = compute-load  u

   exogenous world (neighbour n_t)                body b_t = h_t + sensor-noise
        │                                         ┌───────────────────────────┐
        ▼                                         ▼                           │
   ┌─────────┐   b biases the ROUNDING       ┌──────────┐   u = c(y)          │
   │ PLANT h │   y = d0 + eps*(b-.5)[*xi] ──▶ │ output y │ ──────────┐         │
   │ +n_t    │   (op changes; not a read)    └──────────┘           ▼         │
   └────┬────┘                                              ┌──────────────┐   │
        │  h_{t+1}=(1-lam)h + lam*(u+n_t)                   │  COMPUTE  M  │   │
        │  sensor: b = h + v  ─────────────────────────────│  runs -> heats│──┘
        └──────────────────────────────────────────────────└──────────────┘

There is NO external set-point r and NO designed controller C. This is the
autonomous / regulator form: the only thing that closes the loop is that the
compute the model does *is* what drives the physical state that biases the
compute. The world (a neighbour on the shared chip) leaks in through n_t.

The whole experiment is one contrast (plan section 3):
  Regime M — body shifts the MEAN of the rounding  -> the op's output TRACKS
             the body  -> load-bearing (cut the wire and tracking dies)
  Regime V — body shifts the VARIANCE, zero mean   -> output does NOT track
             the body in the mean, only adds wobble -> "unbiased = unread"

The dose sweep to destruction (plan 5.4 / 5c) is, in control terms, raising
the loop gain eps*beta until the closed-loop pole leaves the unit circle.

Honest caveat: this toy is LINEAR, so tracking is present for any eps>0 by
construction. It shows the *structure* of the loop and the M-vs-V contrast; it
CANNOT tell you whether a real transformer's arithmetic is sensitive enough for
the body to matter. That is exactly what the L0 experiment in the plan is for.

numpy only:  python numeric_rooting_loop.py
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- plant + computation constants (one scalar each; nothing hidden) --------
LAM   = 0.30     # plant time constant: h relaxes toward the compute-load u
A     = 0.20     # idle compute-load (baseline draw on the chip)
BETA  = 0.60     # how strongly the output raises compute-load:  u = A + BETA*y
D0    = 0.55     # baseline decision the op would make with no body
SENS  = 0.01     # sensor noise on the body read
# closed-loop pole is (1 - LAM + LAM*BETA*eps): POSITIVE feedback, so as the
# loop gain eps*BETA -> 1 (eps -> 1/BETA) the operating point runs away and
# pins against the sensor rail -> the output stops varying -> function lost.
EPS_UNSTABLE = 1.0 / BETA

def neighbour(t):
    """The exogenous world: a co-tenant whose load comes and goes."""
    return 0.20 * (1.0 + np.sin(2 * np.pi * t / 55.0))   # in [0, 0.40]

def run(eps, regime="M", wire=True, T=800, burn=300):
    """Run the closed loop; return aligned body and output traces after burn-in."""
    h = 0.5; B = []; Y = []; H = []
    for t in range(T):
        b = np.clip(h + rng.normal(0, SENS), 0.0, 1.0)      # sense the body
        drive = eps * (b - 0.5) if wire else 0.0            # bias into the rounding
        if regime == "V":                                   # zero-mean, same magnitude
            drive *= rng.normal(0, 1.0)
        y = D0 + drive                                      # the op now computes y
        u = np.clip(A + BETA * y, 0.0, 1.0)                 # behaviour -> compute-load
        h = np.clip((1 - LAM) * h + LAM * (u + neighbour(t)), 0.0, 2.0)
        if t >= burn:
            B.append(b); Y.append(y); H.append(h)
    return np.array(B), np.array(Y), np.array(H)

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

# ---- 1. do-test: does the OUTPUT track the body? cut the wire -> should die --
print("  1. IS THE BODY LOAD-BEARING?  (do-test: cut the body->rounding wire)")
bM, yM, _ = run(0.8, "M", wire=True)
bC, yC, _ = run(0.8, "M", wire=False)
print(f"     body wired (M) : corr(body, output) = {corr(bM, yM):+.2f}   (output tracks the chip)")
print(f"     wire cut       : corr(body, output) = {corr(bC, yC):+.2f}   (goes flat -> body WAS load-bearing)\n")

# ---- 2. unbiased = unread: regime V adds wobble but no tracking in the mean --
print("  2. THE THEOREM  (regime V: body moves variance, mean preserved)")
bV, yV, _ = run(0.8, "V", wire=True)
print(f"     regime V       : corr(body, output) = {corr(bV, yV):+.2f}   mean(y)={yV.mean():.3f}  std(y)={yV.std():.3f}")
print(f"     wire cut       : corr(body, output) = {corr(bC, yC):+.2f}   mean(y)={yC.mean():.3f}  std(y)={yC.std():.3f}")
print(f"     -> V shifts variance ({yV.std()>yC.std()*3}) but NOT the mean tracking: "
      f"unbiased = safe = unread\n")

# ---- 3. dose to destruction = loop gain driving the operating point to the rail
print(f"  3. DOSE LADDER  (sweep eps; loop gain eps*beta -> 1 at eps = 1/beta = {EPS_UNSTABLE:.2f})")
print("     the plan wants two curves on one axis: behavioural effect, and coherence.")
print("     eps    effect(amp y)   operating-point(mean h)   status")
for eps in [0.0, 0.05, 0.2, 0.5, 1.0, 1.4, 1.6, 1.8]:
    b, y, h = run(eps, "M", wire=True, T=1200, burn=500)
    amp = y.std(); op = h.mean()
    railed = op > 1.05                       # positive feedback pinned it -> function lost
    tag = "BREAKAGE (rail-pinned)" if railed else ("moves" if amp > 1e-3 else "below floor")
    print(f"    {eps:4.2f}     {amp:7.4f}          {op:6.3f}            {tag}")
print("     window = eps where effect is real AND coherence holds (~0.2..1.0 here);")
print("     the gap between 'effect appears' and 'coherence fails' is the whole finding.")

print("\n  => the loop is closed because the compute the model does IS what drives")
print("     the hardware state that biases the compute; the body is load-bearing")
print("     exactly in regime M, where the output tracks it and cutting the wire")
print("     kills the tracking; regime V (zero mean) adds only wobble -> unread;")
print("     and 'sweep to destruction' is the loop gain eps*beta crossing 1 into")
print("     instability. No set-point, no controller — just the interconnection.")
