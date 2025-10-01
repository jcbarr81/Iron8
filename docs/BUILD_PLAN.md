# Baseball Sim Engine – Build Plan (Codex‑Runnable)

This file is the authoritative plan. Each task block contains:
- **What to build**
- **Why**
- **Acceptance Criteria**
- **Where the code lives**
- **Follow-up tests**

We prioritize the **Plate‑Appearance (PA) outcome model** first, then (optional) **Ball‑in‑Play (BIP)** and **advancement** layers.

---

## Phase A — MVP Service (PA Outcomes + Ratings)

### A1. Project Setup
**What:** Boot a minimal Python project with FastAPI, Pydantic, NumPy, pandas, scikit-learn.  
**Why:** Provide an API for the game to call.  
**Where:** `requirements.txt`, `Makefile`, `src/baseball_sim/*`, `run_server.py`.

**Acceptance Criteria**
- `make setup` installs dependencies.
- `make serve` launches `http://localhost:8000/health` with `{"status":"ok"}`.
- Code passes `pytest -q` with starter tests.

---

### A2. Ratings Adapter (authoritative mapping)
**What:** Convert game ratings (0–99) → normalized, monotone features.  
**Why:** Ratings should consistently shift outcomes the right way.  
**Where:** `src/baseball_sim/features/ratings_adapter.py`, `src/baseball_sim/features/transforms.py`.

**Acceptance Criteria**
- Implements batter mappings: 
  - `gf` → **lower** LA (more GB), `pl` → **pull** spray shift, `sc` → **RISP gate** bonus (on hits), `fa` → defensive OAA-like effect, `arm` → runner advancement resistance.
- Implements pitcher mappings:
  - pitch mix: `fb/sl/si/cu/cb/scb/kn` → “stuff” & GB/air biases,
  - `arm` or `as` → velocity / stuff,
  - `fa` (pitcher fielding) → bunts/comebackers.
- Functions are **pure**, **clamped**, unit-tested with fakes.

---

### A3. PA Outcome Model (Multinomial)
**What:** Train a multinomial classifier over `{BB, K, HBP, IP_OUT, 1B, 2B, 3B, HR}` using:
- Ratings features (adapter output)
- Context features (base/outs, count, handedness, park toggles)
**Why:** Core engine to advance the sim.
**Where:** `src/baseball_sim/models/pa_model.py`, `scripts/train_pa.py`.

**Acceptance Criteria**
- `PaOutcomeModel.fit(X, y)` trains on sample/training data (placeholder dataset file or synthetic).
- `predict_proba(features)` returns well-formed dict of class→probabilities summing to 1.
- Model artifact saved to `artifacts/pa_model.joblib`.

---

### A4. Calibration (Per-class, per-slice)
**What:** Fit isotonic/Platt calibrators per class and optional slices (RISP, parks).  
**Why:** Turn scores into **honest probabilities** (Brier/ECE checks).  
**Where:** `src/baseball_sim/calibration/calibrators.py`, `scripts/calibrate_pa.py`.

**Acceptance Criteria**
- `Calibrator.apply(probs, slice_meta)` returns calibrated probs.
- Reliability tests pass on synthetic eval set.

---

### A5. Serving API
**What:** `POST /v1/sim/plate-appearance` accepts full game state + ratings, returns probabilities (+ sampled event) with deterministic RNG when `seed` is provided.  
**Where:** `src/baseball_sim/serve/api.py`, `src/baseball_sim/schemas.py`, `src/baseball_sim/sampler/rng.py`, `run_server.py`.

**Acceptance Criteria**
- Health check returns ok.
- Example request returns:
  - `model_version`
  - `probs` for all classes
  - `sampled_event`
  - `rng.seed`
- Unit tests validate schema and determinism with a fixed seed.

---

## Phase B — Optional Fidelity

### B1. Ball‑in‑Play (BIP) Generator
**What:** Given `IP_OUT` or `HIT` pre‑outcome, generate EV/LA/spray with shifts from ratings (`ph`↑ EV, `gf`↓ LA, `pl`→ pull).  
**Where:** `src/baseball_sim/models/bip_model.py`.

**Acceptance Criteria**
- `predict(ev, la, spray)` distribution responds monotonically to ratings.
- Config flag to toggle BIP path.

---

### B2. Baserunning & Fielding Advancement
**What:** Convert BBE + runners + defense (`fa`, `arm`) into outs/advances.  
**Where:** `src/baseball_sim/models/advancement_model.py`, `src/baseball_sim/utils/parks.py`.

**Acceptance Criteria**
- Runner send/hold decisions respond to OF arm.
- IF DP conversion responds to `fa`.
- Integrates with /v1/sim/plate-appearance when `return_contact` is true.

---

## Phase C — Export & Performance

### C1. ONNX Export & Runtime
**What:** Export PA model to ONNX and load with ONNX Runtime for CPU speed.  
**Where:** `scripts/export_onnx.py`, `src/baseball_sim/serve/onnx_runtime.py`.

**Acceptance Criteria**
- `make export-onnx` produces `.onnx`.
- Server can run either joblib or ONNX via config switch.

---

## Non-Goals / Guardrails
- Do **not** use demographic fields (e.g., ethnicity/skin_tone) in predictions.
- Ratings are **shifters**, not absolutes; calibration maintains league realism.
