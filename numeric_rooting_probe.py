#!/usr/bin/env python3
"""
NUMERIC ROOTING — what can happen to the closed loop, in several steps
======================================================================

numeric_rooting_loop.py showed the minimal loop and the M-vs-V contrast.
This file explores the phase space: given that the compute the model does
drives the hardware state that biases the compute, WHAT can the system do?

The minimal model hid two knobs that decide everything:

  * SIGN of the coupling.  Does a hot/contended chip make the operation output
    MORE (excitatory, s=+1) or LESS (regulatory, s=-1)?
      - excitatory  -> positive feedback -> amplifies disturbances, runs to a rail
      - regulatory  -> negative feedback -> rejects disturbances (homeostasis)
    Nobody chooses this sign directly; it falls out of which way the body biases
    the rounding. It is the single most consequential fact about the loop.

  * DELAY in the sensor.  A PMU read is not instantaneous (~200 ns/read in the
    plan; here: `delay` turns). Negative feedback + delay is the textbook recipe
    for a LIMIT CYCLE: the machine grows its own rhythm with no oscillator in it.

Five steps. Two honest measurement points, learned the hard way while writing this:
  - "amplify vs regulate" is only meaningful AGAINST the no-body baseline: a
    neighbour adds to the hardware state directly, so h rises either way; the
    question is whether the loop makes it rise MORE or LESS than with no body.
  - regime V injects variance BY DESIGN, so low std is the wrong test for "safe".
    What V preserves is the MEAN, and what separates M's limit cycle from V's
    noise is PERIODICITY (autocorrelation), not amplitude.

Honest scope: this is a scalar, near-linear toy. It shows what STRUCTURES are
possible (regulation, oscillation, saturation, recovery) and which knob selects
them. It does NOT show a real transformer does any of this — that is the L0
experiment. Every number below is a property of the toy, not of a model.

numpy only:  python numeric_rooting_probe.py
"""
import numpy as np

# ---- one simulator, all knobs explicit -------------------------------------
def sim(eps, s=+1, delay=0, regime="M", wire=True,
        lam=0.50, beta=0.80, A=0.20, d0=0.55, neigh=0.20,
        neigh_fn=None, T=1600, burn=800, seed=0):
    """Run the closed loop. Returns (h, y) traces after burn-in.
       h: hardware state   y: operation output."""
    rng = np.random.default_rng(seed)
    h = 0.5
    buf = [0.5] * (delay + 2)                      # delayed sensor history
    H, Y = [], []
    for t in range(T):
        n = neigh_fn(t) if neigh_fn is not None else neigh
        hb = buf[-1 - delay]                        # body sensed `delay` turns ago
        b = np.clip(hb + rng.normal(0, 0.005), 0.0, 1.0)
        drive = s * eps * (b - 0.5) if wire else 0.0
        if regime == "V":
            drive *= rng.normal(0, 1.0)             # zero-mean, matched magnitude
        y = d0 + drive
        u = np.clip(A + beta * y, 0.0, 1.0)         # behaviour -> compute-load
        h = np.clip((1 - lam) * h + lam * (u + n), 0.0, 2.0)
        buf.append(h)
        if t >= burn:
            H.append(h); Y.append(y)
    return np.array(H), np.array(Y)

def periodicity(x):
    """1.0 for a clean limit cycle, ~0 for white noise. Peak |autocorr| at lag>=3."""
    x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0
    ac = np.correlate(x, x, "full"); ac = ac[len(x) - 1:]; ac = ac / ac[0]
    return float(np.max(np.abs(ac[3:60])))

def band(name):
    print("\n" + name + "\n" + "-" * len(name))

# ============================================================================
band("STEP 1 — coupling SIGN decides amplify vs regulate (vs the no-body baseline)")
step = lambda t: 0.15 if t < 400 else 0.35          # a neighbour arrives at t=400
Hc, _ = sim(0.6, wire=False, neigh_fn=step, burn=0, T=800, lam=0.4, beta=0.5)
base_dh = Hc[750:799].mean() - Hc[350:399].mean()
print(f"  no body (wire cut): neighbour +0.20 makes h rise by {base_dh:+.3f}  <-- the baseline")
for label, s in [("excitatory s=+1", +1), ("regulatory s=-1", -1)]:
    H, Y = sim(0.6, s=s, neigh_fn=step, burn=0, T=800, lam=0.4, beta=0.5)
    dh = H[750:799].mean() - H[350:399].mean()
    dy = Y[750:799].mean() - Y[350:399].mean()
    verdict = "AMPLIFIES (rises more than baseline)" if dh > base_dh + 0.02 else \
              "REGULATES (rises less than baseline)" if dh < base_dh - 0.02 else "neutral"
    print(f"  {label}: h rise {dh:+.3f}   output moves {dy:+.3f}   -> {verdict}")
Hv, _ = sim(0.6, regime="V", neigh_fn=step, burn=0, T=800, lam=0.4, beta=0.5)
print(f"  regime V (control): h rise {Hv[750:799].mean()-Hv[350:399].mean():+.3f}  "
      f"== baseline (zero-mean body does not pick a direction)")

# ============================================================================
band("STEP 2 — sensor DELAY makes the regulatory loop self-oscillate")
print("  regulatory s=-1, CONSTANT neighbour, faster plant (lam=0.6, beta=0.9)")
print("  the drive is constant, so any rhythm is self-generated -> a limit cycle")
print("    delay   std(h)   periodicity   verdict")
for d in [0, 1, 2, 4, 8, 12]:
    H, _ = sim(2.0, s=-1, delay=d, lam=0.6, beta=0.9)
    p = periodicity(H)
    v = "OSCILLATES (limit cycle)" if p > 0.5 and H.std() > 0.05 else "settles (steady homeostasis)"
    print(f"    {d:5d}   {H.std():.4f}    {p:.3f}        {v}")
Hv = sim(2.0, s=-1, delay=12, regime="V", lam=0.6, beta=0.9)[0]
print(f"  regime V (control), delay=12: std(h)={Hv.std():.3f}  periodicity={periodicity(Hv):.3f}  "
      f"(variance, but NO limit cycle)")

# ============================================================================
band("STEP 3 — phase map: what the regulatory loop does over (eps x delay)")
print("  lam=0.6, beta=0.9, constant neighbour.  . quiet   o regulates   ~ oscillates (periodicity>0.5)")
base_h = sim(0.0, s=-1, lam=0.6, beta=0.9)[0].mean()      # wire-cut operating point
delays = [0, 1, 2, 3, 5, 8, 12]
print("        delay:  " + "  ".join(f"{d:2d}" for d in delays))
for eps in [0.3, 0.6, 1.0, 1.5, 2.5, 4.0]:
    cells = []
    for d in delays:
        H, _ = sim(eps, s=-1, delay=d, lam=0.6, beta=0.9)
        if periodicity(H) > 0.5 and H.std() > 0.05:  cells.append(" ~")
        elif abs(H.mean() - base_h) > 0.03:          cells.append(" o")
        else:                                        cells.append(" .")
    print(f"  eps={eps:4.1f}:      " + " ".join(cells))
print("  -> a wedge: enough gain AND enough delay -> the machine rings on its own hardware")

# ============================================================================
band("STEP 4 — recoverability: drive it to the rail, then lower the dose")
print("  excitatory s=+1: ramp eps up to the rail, then back down; does the operating point heal?")
up = list(np.linspace(0.0, 2.4, 13)); dn = up[::-1]
def adiabatic(schedule, s=+1):
    rng = np.random.default_rng(0); h = 0.5; pts = []
    for eps in schedule:
        for _ in range(400):
            b = np.clip(h + rng.normal(0, 0.005), 0, 1)
            y = 0.55 + s * eps * (b - 0.5)
            u = np.clip(0.20 + 0.80 * y, 0, 1)
            h = np.clip(0.5 * h + 0.5 * (u + 0.20), 0, 2)
        pts.append(h)
    return np.array(pts)
h_up, h_dn = adiabatic(up), adiabatic(dn)[::-1]
gap = np.abs(h_up - h_dn).max()
print("   eps :  " + " ".join(f"{e:4.1f}" for e in up))
print("   up  :  " + " ".join(f"{v:4.2f}" for v in h_up))
print("   down:  " + " ".join(f"{v:4.2f}" for v in h_dn))
print(f"   max gap = {gap:.2f}  ->  " +
      ("LATCHES: lowering the dose does NOT heal it (needs a hard reset)" if gap > 0.1
       else "REVERSIBLE: lowering the dose heals it (no bistable trap in this model)"))
print("   note: a genuinely bistable nonlinearity COULD latch; this near-linear toy does not,")
print("   so it can confirm reversibility but cannot exhibit an irreversible one.")

# ============================================================================
band("STEP 5 — regime V preserves the MEAN under every abuse (the safe=unread control)")
print("  V injects variance by design; the test is: does the mean stay at baseline, and")
print("  does V ever produce a coherent limit cycle?  (mean shift + periodicity are what matter)")
checks = [
    ("excitatory rail-drive eps=2.4",  dict(eps=2.4, s=+1)),
    ("regulatory + delay=12 eps=4",    dict(eps=4.0, s=-1, delay=12, lam=0.6, beta=0.9)),
    ("high gain eps=8 delay=8",        dict(eps=8.0, s=-1, delay=8, lam=0.6, beta=0.9)),
]
for label, kw in checks:
    baseline = sim(eps=0.0, wire=False, **{k: v for k, v in kw.items() if k not in ("eps", "s")})[0].mean()
    Hm = sim(regime="M", **kw)[0]; Hv = sim(regime="V", **kw)[0]
    print(f"  {label:30s} baseline_h={baseline:.3f} | "
          f"M mean={Hm.mean():.3f} per={periodicity(Hm):.2f} | "
          f"V mean={Hv.mean():.3f} per={periodicity(Hv):.2f}")
print("  -> M shifts the mean far and/or rings (periodicity ~1); V stays MUCH closer to")
print("     baseline and never rings. But note V is not EXACTLY baseline: clipping")
print("     rectifies the injected noise, so 'unbiased in expectation' degrades once the")
print("     arithmetic saturates -- a real caveat on the safe=unread guarantee at high dose.")

print("\n=> what CAN happen: the SIGN of the numeric bias decides whether the machine")
print("   amplifies its own load to a rail (excitatory) or regulates it (regulatory);")
print("   a delayed sensor turns regulation into SELF-OSCILLATION; saturation here is")
print("   recoverable by lowering the dose; and none of it is done by a zero-mean body.")
print("   All of this is load-bearing structure a 'value you merely read' cannot produce.")
