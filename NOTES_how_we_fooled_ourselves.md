# How we fooled ourselves

Every entry below is a result we believed, wrote down, and then killed. They are more useful than our positives. Roughly chronological.

---

### 1. The positive control that voided a month of nulls

We had a stack of clean negatives — the body doesn't move the policy, the self-subspace has no causal effect, the workspace shows nothing. Then we ran the obvious check we had skipped: **can our write channel inject a concept the model demonstrably knows, and have it detect that?**

It could not. **z = 1.14.** The instrument didn't work.

Every null gathered with it was uninterpretable — we had never established that our injection did anything at all. After repair (single-position injection instead of all-position, calibrated magnitude, letting a thinking model actually think, many-sample concept vectors) the same control passes at **z = 26.7**.

**Rule adopted:** no negative result may be reported unless a positive control on the same channel, in the same session, has passed. *A null from a broken instrument is not a finding.*

---

### 2. Five identical conditions scored higher than five different ones

We wanted to show an agent's actions were physically separable — that it had several independent ways to affect its own body. We measured the effective rank of the action→body effect matrix. Pre-registered threshold ≥ 2.0. We got **3.227 of a possible 4.0**. All thresholds passed.

Then the adversarial pass ran the identical pipeline on conditions that could not possibly differ:

| Condition | Effective rank |
|---|---|
| Five copies of `rest` (sleep) | **3.451** |
| Five copies of `compute` | 2.564 |
| **Our five distinct actions** | **3.227** |
| Gaussian noise, 20 000 draws | 3.577 ± 0.190 |

Five identical sleeps scored *higher* than five different actions. The statistic was sampling noise on a small matrix. Result struck.

**Rule adopted:** any statistic used to argue that things *differ* must first be run on N copies of a **single** condition and on matched noise.

---

### 3. A threshold that was unreachable by construction

Earlier version of the same experiment: three actions, pre-registered rank ≥ 2.0, measured **1.996**. Reported as failure.

But mean-centering three rows caps the rank at exactly 2.0. We had measured **99.8 % of the maximum possible** and called it a null. Not a post-hoc threshold — a pre-registered *impossible* one.

**Rule adopted:** compute the ceiling of your metric under your own design before registering a threshold against it.

---

### 4. Six meters that couldn't move

In one night, six failures of a single family: *something reported as done that did not happen.*

- An action logged as "running compute" while the GPU sat at **0 % utilization and 47 °C**.
- `/proc/stat` fields read as levels. They are **cumulative counters since boot** — they look frozen unless differenced over a known interval.
- I/O actions issued against **tmpfs**, which lives in RAM. The disk was never touched; the disk meter was correctly reporting nothing.
- Firmware-averaged sensors asked to resolve bursts shorter than their own averaging window.
- Instantaneous samplers aliasing between compute batches.

**Rule adopted:** before a channel enters the body vector, a scripted stimulus must move it by a pre-registered margin in SDs of its own idle noise — and its absence must leave it flat. Before an action enters the repertoire, it must show a signed effect on its own organ *and* an independent execution witness. **A meter that has never been seen to move is not evidence of calm.**

---

### 5. Twelve channels, 2.57 dimensions

We were pleased with a rich multi-channel body. Then we looked at the correlation matrix: `gpu_temp ~ gpu_power` at **r = 0.992**, and the whole retained set spanned only ~2.57 independent directions.

We had five views of one heat curve, not five organs. On one die, temperature *is* power with a lag — you cannot buy independence by reading more numbers off the same subsystem.

**Rule adopted:** publish the correlation matrix and participation ratio with every body-vector revision. Channels above |r| ≈ 0.95 count as one.

---

### 6. We built the shortcut ourselves

We kept trying to force the body signal to route through the model's self-representation, and kept finding it took some other path. Then the geometry was pointed out to us: **a linear low-rank direction that survives to the final layer is, by construction, readable by the unembedding.** We had built a straight pipe from sensor to mouth and then asked gradient descent politely not to use it. No loss term can prevent that.

The fix is topological, not an objective: sever the sensor from the report head; let the body enter only through a small module that is forced to be a **forward self-model** — predicting its own next body state and its own next outcome. A passthrough contains no future, so prediction cannot be faked by relaying.

---

### 7. The report that learned to say the right thing

Training the coupling with a loss on report tokens produced a beautiful **0.97 correlation** between the model's stated state and its real one — with **zero mediation**. It had learned to say the right words with nothing routed through the self. An actor hitting a cue, not a state being read.

**Rule adopted:** train on task outcome under physical constraint only. Never on a detection loss. Never on report tokens.

---

### 8. Homeostasis that survived cutting the sensor

Our regulation loop looked healthy — the agent returned toward its set-point on 14 of 15 channels. Then the sensor-cut control: return with the real sensor **3.171**, with it cut **3.275**. Cutting the body made regulation *slightly better*.

It was regulating a body the *prompt* implied, not the one it had. We had flagged this exact failure mode as the program's #1 false positive months earlier, and then committed it.

---

### 9. Continuous projection is worse than no body

We grafted the body in as a continuous projected vector — three separate times. Published measurements on an analogous setup: tokenized state 4.48 > **no state at all 4.44** ≈ MLP→action-head 4.44 > **MLP→continuous input 4.15**.

The continuous graft is *pseudo-aligned*: fully in-distribution, fully decodable, computationally inert. The projector moves the distribution's centroid into token space, not its manifold. A robotics group (π₀.₅) deleted their continuous state projector and discretized state into text tokens instead.

**Rule adopted:** encoding format is an open variable to be tested, never a default.

---

### 10. Seven months of frozen backbone

We kept the base model frozen throughout, and repeatedly measured that it did not learn to read the body. It was never permitted to. The absence of an ability we had forbidden is not a finding about the model.

---

## The pattern

Nine of the ten are the same shape: **we measured something we had installed, or failed to measure something we had prevented, and did not run the control that would have told us which.**

The two habits that catch it: (a) *always run the null condition your statistic would score on nothing*, and (b) *validate the instrument before trusting its silence.*
