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

### 11. An ignition result that was an algebraic identity

We injected a unit direction into the residual stream at layer 52 with scale α, read the projection onto the verbalizable-workspace cone, and divided by α. The gain came out flat — 537.3 at α=0.125 and 537.3 at α=6 — and we concluded that entry into the workspace is graded rather than all-or-none.

The residual stream has an identity skip. The injection scale is the mean residual norm, **575**. The cone fraction of our direction was **0.9326**. Their product is **536.2**, against a measured **537.3** — agreement to 0.2 %. A twenty-line synthetic with no network in it at all reproduces the curve with relative range **0.000000**. Our reported 3 % variation *was the entire contribution of a 32-billion-parameter model.*

Two further defects, both ours:

- **The threshold could not be passed.** Our bar required the network to spontaneously add an in-cone component ~0.47× the norm of the whole injected vector. A *planted*, textbook, unmistakable 50 % all-or-none amplification scores **FAIL** on our rig. We pre-registered a bar roughly 16× above the achievable signal and reported the miss as a fact about the model.
- **A competition result was a dictionary-budget artifact.** Our sub-additivity test used a fixed k=10 sparse fit, and each injected direction was *built from 10 atoms* — so a two-direction mixture had to be fit with half the budget it needed. The no-network synthetic gives 0.875 / 0.966 / 1.255 for matched-in-cone / matched-random / unmatched; we measured 0.827 / 0.990 / 1.149. The competition was for slots in our own solver.

And what we had banked as the strongest evidence was the tell: two runs agreeing to **1.4 %** across different prompt counts, different α grids and different direction construction. Real neural measurements do not replicate that well under changed prompts. Algebraic identities do.

**Fix, one line:** subtract the known pass-through before projecting — `‖P_cone(δ − α·scale·u)‖/α`. On the same planted signals that scores 88.8 instead of 0.032. Three orders of magnitude of sensitivity.

**Rule adopted:** suspiciously good replication is evidence of a tautology, not of a fact. And every injection-based readout is checked against its analytic pass-through before it is believed.

---

### 12. "Real hardware wired to decisions" was a labelled flashcard

We trained a real-weight edit on live GB10 telemetry — joules, temperature, clock — and measured that inducing genuine load made the model choose the protective action. Closed loop on fresh readings: **0.989**. Reverting the edit: 0.14. Forgetting: KL 0.002.

The normaliser subtracts the midpoint of the two training readings and divides by their difference. Substituting the only two training inputs gives **(−0.5, −0.5, +0.5)** and **(+0.5, +0.5, −0.5)** — antipodal, *for any hardware values whatsoever*. The model never saw more than one bit, and the "shift" is the training accuracy of a two-point classifier on its own two points.

Worse in detail:
- The code sleeps 300 ms *after* the load completes, so the "hot" power reading is the cooldown. Normalised it is **−0.07** — the joule channel, the headline organ of the whole program, sat on the decision boundary carrying nothing. The bit rode entirely on thermal lag.
- Evaluation used the exact task strings and word pairs seen in training. Zero held-out prompts.
- The forgetting number was measured on the same two prompts used as the KL anchor *inside the training loss*.
- The coherence check ran at the mean body value, which normalises to the zero vector — coherence was verified with no injection at all.

**Controls now mandatory for any body-wiring claim.** *Synthetic body*: replace the sensor with a coin flip; if the number survives, the claim is about the wire. *Crosswire*: keep the machine genuinely idle and inject the stored "hot" reading; if the decision still flips, the loop is the script's, not the model's.

---

### 13. A recovery number inside the free lunch

We lesioned half the weights of four MLP output matrices, forbade editing them, and healed by self-supervised training that grew new connections around the damage — 80.4 % recovery on held-out text, and reverting the new connections restored the damage.

The whole lesion costs **0.139 nats**. Published work (the "Hydra effect") shows transformers spontaneously recover ~70 % of an ablation with **zero training**, because downstream layers route around it. Our heal corpus was eight generic sentences, evaluated on five held sentences of the same register. Until the sham-heal arm runs — identical steps on an *undamaged* model — nothing distinguishes function-specific repair from getting better at encyclopedic English.

Also: the "2,036,858 grown connections", identical across all four layers, is `int(0.03 × 3584 × 18944)` — a fixed density budget, not an emergent count.

**What survived, and it is the useful negative:** across four increasingly damaged states, CKA reads **0.9999, 0.9997, 0.9996, 0.9993** while task accuracy falls from 0.944 to 0.778. It is flat to four decimal places through a quarter of the capability draining away, and collapses only once the model is already destroyed. **CKA alarms after the catastrophe and cannot be the online recoverability gate the continual-learning literature proposes.** The blunt displacement measure we already use moves 160× in the same window. (Power warning belongs to the claim: n = 7, p ≈ 0.2 — the design cannot resolve the margin, so this is "no support", not "refuted".)

---

## The pattern

Nine of the first ten are the same shape: **we measured something we had installed, or failed to measure something we had prevented, and did not run the control that would have told us which.**

Eleven through thirteen add a second shape, and it is the more embarrassing one: **we measured our own instrument and read the answer as a fact about the model.** An identity skip carrying an injected vector, a normaliser collapsing any two readings to one bit, a solver running out of dictionary slots.

The habits that catch both: (a) *always run the null condition your statistic would score on nothing*; (b) *validate the instrument before trusting its silence*; (c) *before believing a number, compute what it would be if the model were removed entirely*; and (d) *treat a suspiciously exact replication as a warning, not a confirmation*.

Three of these were found by asking someone outside the project to read the code and try to break it. That is the cheapest habit on this list and the one we adopted last.
