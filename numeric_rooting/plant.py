#!/usr/bin/env python3
"""
Multi-timescale hardware plant (the body).

Separation of timescales (Molka/Schöne PMU contention; thermal RC + DVFS):
  contention c  ~ us   -> treated as quasi-static (algebraic)
  power/clock   ~ ms   -> integrated
  thermal  T    ~ s    -> integrated (the slow integrator)

Real closed paths kept:
  L -> P -> T -> f_cap -> f -> P          (thermal throttle, negative)
  P_leak(T) = k_leak*exp(T/T_leak)        (leakage vs temperature, POSITIVE fb)
  L,L_neigh -> c -> IPC_eff               (contention degrades effective throughput)

Channels exposed to the sensor: T (thermal), c (contention), P (power).
The body is coupled back as a precision-weighted homeostatic ERROR, not raw
telemetry (Keramati/Gutkin HRRL; Nave et al. active-inference interoception).
"""
import numpy as np


class Plant:
    def __init__(self,
                 T_amb=40.0, T_set=72.0, T_trip=95.0, T_leak=45.0,
                 R_th=0.85, tau_th=8.0, tau_dvfs=1e-3,
                 f_max=1.0, g_th=0.03, k_dyn=95.0, k_leak=0.5,
                 k_c=0.6, thr=0.4, a_act=1.0, k_neigh=55.0):
        self.p = dict(T_amb=T_amb, T_set=T_set, T_trip=T_trip, T_leak=T_leak,
                      R_th=R_th, tau_th=tau_th, tau_dvfs=tau_dvfs, f_max=f_max,
                      g_th=g_th, k_dyn=k_dyn, k_leak=k_leak, k_c=k_c, thr=thr,
                      a_act=a_act, k_neigh=k_neigh)
        self.T = T_amb + 20.0
        self.f = f_max
        self.P = 0.0
        self.c = 0.0

    def step(self, L, L_neigh=0.0, dt=1e-3, substeps_thermal=True):
        """Advance one control step (dt ~ 1 ms). L, L_neigh in [0,1]."""
        p = self.p
        # FAST: contention quasi-static
        self.c = np.clip(p["k_c"] * (L + L_neigh) - p["thr"], 0.0, 1.0)
        IPC_eff = 1.0 - self.c
        # MEDIUM: DVFS clock chases a thermal-capped target
        f_cap = p["f_max"] - p["g_th"] * max(0.0, self.T - p["T_trip"])
        f_target = min(p["f_max"] * (0.3 + 0.7 * L), f_cap)
        self.f += min(1.0, dt / p["tau_dvfs"]) * (f_target - self.f)  # clamp: large-dt safe
        self.f = float(np.clip(self.f, 0.05, p["f_max"]))
        # power: dynamic (only non-stalled active work) + temperature-dependent leakage
        P_dyn = p["k_dyn"] * self.f ** 3
        P_leak = p["k_leak"] * np.exp(self.T / p["T_leak"])
        self.P = p["a_act"] * L * IPC_eff * P_dyn + P_leak
        # SLOW: thermal RC relaxes toward ambient + (own + co-tenant) power * R_th.
        # The neighbour shares the thermal domain -> it heats the die (exogenous body drive).
        P_total = self.P + p["k_neigh"] * L_neigh
        self.T += min(1.0, dt / p["tau_th"]) * ((p["T_amb"] + P_total * p["R_th"]) - self.T)
        self.T = float(np.clip(self.T, p["T_amb"], 150.0))
        return self.sense()

    def sense(self):
        """Raw channel readings (what a PMU/thermal sensor sees)."""
        return dict(T=self.T, c=self.c, P=self.P, f=self.f)

    def homeostatic_error(self, pi_T=1.0):
        """Precision-weighted thermal deviation from set-point (the coupling signal)."""
        return pi_T * (self.T - self.p["T_set"])


if __name__ == "__main__":
    # step response: load jumps, watch thermal rise, throttle engage, leakage
    pl = Plant()
    log = []
    for t in range(20000):                       # 20 s at dt=1ms
        L = 0.2 if t < 4000 else 0.95
        s = pl.step(L, L_neigh=0.1, dt=1e-3)
        if t % 2000 == 0:
            log.append((t / 1000, s["T"], s["c"], s["P"], s["f"]))
    print("  t(s)   T(C)    c      P(W)   f")
    for r in log:
        print(f"  {r[0]:4.0f}  {r[1]:6.2f}  {r[2]:.3f}  {r[3]:6.2f}  {r[4]:.3f}")
    print(f"  homeostatic error at end: {pl.homeostatic_error():+.2f} C above set-point")
