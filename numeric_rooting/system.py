#!/usr/bin/env python3
"""
The closed loop, at two fidelities.

ABSTRACT (abstract_loop): the reduced delayed map from control theory,
    x_{t+1} = a*x_t + sign*K*sat(x_{t-d}) + noise
where x is the sensed homeostatic error, K bundles all loop gains, d the
sensor+plant delay, a the plant's natural relaxation, sat the arithmetic/
activation saturation. Cheap -> dense (K,d) cartography, bifurcation, Lyapunov.

CONCRETE (ConcreteLoop): a small fixed MLP whose every matmul uses body-biased
stochastic rounding, driving the real multi-timescale Plant, closing the loop:
    body error e_t -> rounding bias -> network output y -> compute-load L
                   -> plant -> temperature -> e_{t+1}
Two arithmetic regimes, both INSIDE the rounding, both body-driven:
    M : mean-shift knob b = sign*kappa*e_sensed         (changes the function)
    V : precision m_eff = m_base - gamma*|e_sensed|      (variance only; unbiased)
'cut' severs the wire (b=0, m_eff=m_base) as the do-test control.
"""
import numpy as np
from .srr import sr_round
from .plant import Plant


# ---------------- abstract reduced loop ----------------
def sat(x, slope=1.0, delta=1.0):
    return np.clip(slope * x, -delta, delta)


def abstract_loop(K, d, a=1.0, sign=-1, slope=1.0, delta=1.0,
                  noise=0.0, T=4000, burn=1500, seed=0):
    """Return the post-burn bias series of x_{t+1}=a*x_t + sign*K*sat(x_{t-d})."""
    rng = np.random.default_rng(seed)
    buf = [0.05] * (d + 1)                     # small seed perturbation
    x = 0.05
    out = []
    for t in range(T):
        xd = buf[-1 - d]
        x = a * x + sign * K * sat(xd, slope, delta)
        if noise:
            x += noise * rng.standard_normal()
        x = float(np.clip(x, -50, 50))         # numeric guard (divergence flag handles rails)
        buf.append(x)
        if t >= burn:
            out.append(x)
    return np.array(out)


# ---------------- concrete body-biased-arithmetic loop ----------------
class ConcreteLoop:
    def __init__(self, K=48, H=32, B=16, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.H, self.B = K, H, B
        self.X = rng.standard_normal((B, K)) * 0.3        # fixed input batch
        self.W1 = rng.standard_normal((K, H)) / np.sqrt(K)
        self.W2 = rng.standard_normal((H, H)) / np.sqrt(H)
        self.w3 = rng.standard_normal(H) / np.sqrt(H)

    def _sr_matmul(self, Xin, W, m, b, rng):
        """Xin:(B,Kin) W:(Kin,Hout) -> (B,Hout), SR after each accumulation step."""
        Bn = Xin.shape[0]; Kin, Hout = W.shape
        acc = np.zeros((Bn, Hout))
        for k in range(Kin):
            acc = sr_round(acc + Xin[:, k][:, None] * W[k][None, :],
                           m, rng.random((Bn, Hout)), b)
        return acc

    def forward(self, m, b, rng):
        h1 = np.maximum(0.0, self._sr_matmul(self.X, self.W1, m, b, rng))
        h2 = np.maximum(0.0, self._sr_matmul(h1, self.W2, m, b, rng))
        y = self._sr_matmul(h2, self.w3[:, None], m, b, rng)[:, 0]
        return float(y.mean()), h2                        # scalar output, hidden acts

    def run(self, regime="M", sign=-1, kappa=0.02, delay=0,
            m_base=7, gamma=0.6, neigh_fn=None, gain=None,
            T_steps=2500, burn=800, seed=1, dt=0.05):
        rng = np.random.default_rng(seed)
        pl = Plant()
        # calibrate output->load gain so the bias range spans a useful load range
        y0, _ = self.forward(m_base, 0.0, np.random.default_rng(99))
        if gain is None:
            y1, _ = self.forward(m_base, min(0.5, abs(kappa) * 25), np.random.default_rng(98))
            dy = y1 - y0
            gain = 0.6 / dy if abs(dy) > 1e-6 else 0.0     # 0 -> arithmetic too weak to steer
        buf = [0.0] * (delay + 1)
        bias_s, e_s, T_s, L_s, y_s, var_s = [], [], [], [], [], []
        logvar_feat = []
        for t in range(T_steps):
            e_sensed = buf[-1 - delay]
            if regime == "M":
                b = sign * kappa * e_sensed; m_eff = m_base
            elif regime == "V":
                # body sets how many mantissa bits survive -> variance only, mean unbiased.
                # gamma is the V-dose (bits dropped per degree of homeostatic error).
                b = 0.0
                m_eff = int(np.clip(round(m_base - gamma * abs(e_sensed)), 2, m_base))
            else:  # cut
                b = 0.0; m_eff = m_base
            y, h2 = self.forward(m_eff, b, rng)
            L = float(np.clip(0.5 + gain * (y - y0), 0.05, 0.98))
            neigh = neigh_fn(t) if neigh_fn else 0.10
            pl.step(L, neigh, dt=dt)
            e = pl.homeostatic_error()
            buf.append(e)
            if t >= burn:
                bias_s.append(b); e_s.append(e); T_s.append(pl.T)
                L_s.append(L); y_s.append(y); var_s.append(float(h2.var()))
                logvar_feat.append(np.log(h2.var(axis=0) + 1e-6))   # per-unit log-var
        return dict(bias=np.array(bias_s), e=np.array(e_s), T=np.array(T_s),
                    L=np.array(L_s), y=np.array(y_s), hvar=np.array(var_s),
                    logvar=np.array(logvar_feat), gain=gain, y0=y0)


if __name__ == "__main__":
    from .metrics import periodicity, dominant_period, k_crit, predicted_period
    print("ABSTRACT loop — regulatory, sweep delay near stability boundary")
    for d in [0, 1, 2, 4, 8]:
        Kc = k_crit(d)
        x = abstract_loop(K=1.2 * Kc, d=d, sign=-1)
        print(f"  d={d:2d} K=1.2*Kc={1.2*Kc:.3f}: std={x.std():.3f} per={periodicity(x):.2f} "
              f"period={dominant_period(x)} (theory T*={predicted_period(d)}) "
              f"{'OSC' if periodicity(x) > 0.5 else 'stable'}")
    print("\nCONCRETE loop — one regulatory run with real SR arithmetic")
    cl = ConcreteLoop(seed=0)
    r = cl.run(regime="M", sign=-1, kappa=0.03, delay=2, T_steps=1500, burn=500)
    print(f"  gain={r['gain']:.2f}  T range [{r['T'].min():.1f},{r['T'].max():.1f}]C  "
          f"L mean {r['L'].mean():.2f}  bias|.|mean {np.abs(r['bias']).mean():.4f}  "
          f"periodicity(T)={periodicity(r['T']):.2f}")
