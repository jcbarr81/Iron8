**Scope**
- This list captures remaining work only; completed items include: repo scaffold, ratings adapter, context features (count/base/outs/TTO), players.csv loader, serving API with seeded sampling, synthetic training + artifact I/O stubs, identity/Platt calibrators, and RISP gating (implemented and tested).
- Update: Ball-In-Play generation and fielding/outs handling are REQUIRED (not optional) in this plan.

Order of execution (recommended)
1) Feature Expansion (handedness matchup, park/rules features)
2) Ball-In-Play (BIP) Generator (required)
3) Fielding & Outs Conversion (required)
4) Baserunning & Advancement (required)
5) Steals, Leads, Pickoffs, WP/PB (required)
6) Errors & Official Scoring (required)
7) Defensive Positioning & Strategy (required)
8) Outputs & Telemetry (required)
9) Pitch-By-Pitch Engine (required)
10) Umpires, Zone, Replay (required)
11) Fatigue, Warm-up, Three-Batter Minimum (required)
12) Manager/AI Strategy (required)
13) Weather & Park Dynamics (required)
14) ETL & Training (real data)
15) Calibration (held-out)
16) Evaluation & Reporting
17) ONNX Export & Runtime (performance)
18) Integration Enhancements (players cache reload)
19) Data Sources & Licensing (complete docs/notice)
20) Season/League Simulation (optional)
21) Injury Model & Recovery (optional)

—

1) Feature Expansion
- Handedness matchup feature
  - Acceptance Criteria:
    - Add `ctx_matchup_is_platoon_adv` based on batter handedness vs pitcher throws.
    - Unit test ensures feature value matches handedness inputs.

- Park and rules features
  - Acceptance Criteria:
    - Load `config/parks.yaml` and surface simple directional hints into features.
    - Add `rules.shift_restrictions` feature; tests confirm parsing and inclusion.
    - Features are optional; service runs if config data is absent.

2) Ball-In-Play (BIP) Generator (Required)
- Generate EV/LA/spray and contact type
  - Acceptance Criteria:
    - Implement `src/baseball_sim/models/bip_model.py` to sample EV/LA/spray and `contact_type` in {GB, LD, FB, PU} with monotone shifts: `ph` increases EV; `gf` increases GB share (lower LA); `pl` pulls spray by handedness; `pit_stuff_z` modestly reduces EV.
    - API returns `bip_detail` whenever `return_contact=true` (no config flag gating).
    - Unit tests verify monotonicity for `ph` (EV up), `gf` (more GB), `pl` (spray pulls by handedness).

3) Fielding & Outs Conversion (Required)
- Convert BIP to outs/hits and out subtypes using defense, park, and rules
  - Acceptance Criteria:
    - Implement a fielding conversion layer (in `advancement_model.py` or a new `fielding.py`) that maps (EV, LA, spray, contact_type, runners, outs) + defense to: out_subtype in {GB_OUT, FB_OUT, PU_OUT, LD_OUT} or hit outcome, plus fielder assignment.
    - Use `defense` input (e.g., `oaa_by_pos`) when provided; otherwise fallback to player `fa` by primary position, with pitcher `fa` affecting bunts/comebackers. Include `arm` effects for assists/holds on OF plays.
    - Apply infield-fly rule for high LA, low EV pop-ups with <2 outs and runners on; treat as automatic out without advancement except forced.
    - Double play logic: with runner on 1B and <2 outs, GB to IF computes DP probability increased by IF `fa`; update outs and bases accordingly when DP occurs.
    - API includes `bip_detail.out_subtype`, `bip_detail.fielder`, and DP flag when applicable; `advancement` reflects `outs_recorded` and final `bases_ending`.
    - Unit tests: higher OF `fa` increases FB/LD out conversion; higher IF `fa` increases GB out/DP conversion; pitcher `fa` reduces bunt hit rate.

4) Baserunning & Advancement (Required)
- Runner advancement decisions and scoring on BIP
  - Acceptance Criteria:
    - Implement `src/baseball_sim/models/advancement_model.py` to compute sends/holds/outs using BIP + base state + defense (`fa`, `arm`) and park context.
    - API includes `advancement` whenever `return_contact=true`, with fields: `outs_recorded`, `rbi`, `bases_ending` mapping, and optional `dp` flag.
    - Unit tests: stronger OF `arm` reduces 3B→home sends on medium/short flies; higher IF `fa` increases DP conversions on GB with <2 outs; faster runner (use batter/runner speed if available) improves extra-base advancement.

5) Steals, Leads, Pickoffs, WP/PB (Required)
- Micro-events independent of balls in play
  - Acceptance Criteria:
    - Attempt model for SB/CS using runner speed/steal skill vs pitcher hold_runner and catcher arm.
    - Outcomes include: SB, CS, PO (pickoff), WP, PB; update bases and outs accordingly.
    - Unit tests: higher runner speed -> higher SB rate; higher catcher arm/hold_runner -> higher CS/PO; lower catcher fielding -> higher PB.

6) Errors & Official Scoring (Required)
- Errors, fielder’s choice, sacrifices, earned/unearned runs
  - Acceptance Criteria:
    - Add error model by fielder and play type to produce E, FC, ROE; distinguish SF and SH; award SB vs defensive indifference on steals.
    - Maintain earned vs unearned run accounting and display in outputs.
    - Tests: higher fielder fa -> lower error rate; ROE increases when IF fa is low on GB; SF credited on flyouts with RBI and <2 outs.

7) Defensive Positioning & Strategy (Required)
- Alignments and depth affect outcomes
  - Acceptance Criteria:
    - Support infield alignment: in, double-play depth, corners in, normal; outfield: shallow/normal/deep; shifts constrained by rules.shift_restrictions.
    - Inputs may be provided per PA; default heuristic alignment based on state.
    - Tests: infield in reduces single probability but allows more line-drive hits; outfield deep reduces extra-base hits but increases singles on short flies.

8) Outputs & Telemetry (Required)
- Box scores, play-by-play, win probability
  - Acceptance Criteria:
    - Produce play-by-play log with seeds and trace_id; compute box score and pitcher lines; track earned runs.
    - Add win probability and leverage index based on state engine; expose `/v1/game/boxscore` and `/v1/game/log` for completed sims.
    - Tests: totals match event log; W/L/SV/HLD derived from final state.

9) Pitch-By-Pitch Engine (Required)
- Model count evolution, pitch selection, and pre-pitch events
  - Acceptance Criteria:
    - Add `/v1/sim/pitch` with features: count, batter/pitcher handedness, pitch mix, fatigue/TTO, base state.
    - Returns: pitch_type, called result {ball, called_strike, swing_miss, foul, in_play, hbp}, next_count, and if in_play then delegates to BIP -> fielding -> advancement.
    - Includes pickoff attempts and step-offs with outcomes {no play, PO, balk} gated by rules.
    - Unit tests verify realistic count transitions (e.g., more balls early for wild pitchers), monotone effects of control/stuff, and determinism by seed.

10) Umpires, Zone, Replay (Required)
- Variability in calls and challenges
  - Acceptance Criteria:
    - Per-umpire profile (or generic variance) affecting called strike probability by location; seed-driven randomness.
    - Optional replay/challenge toggles with simple overturn rates; pitch timer/disengagement limits enforced.
    - Tests: different ump profiles yield different called-strike rates at edge; challenge can flip a small fraction of close plays.

11) Fatigue, Warm-up, and Three-Batter Minimum (Required)
- In-game stamina and substitution rules
  - Acceptance Criteria:
    - Model fatigue per pitcher by pitch count and TTO; performance degrades as fatigue rises.
    - Enforce three-batter minimum and warm-up penalty if a reliever enters cold.
    - Tests: rising pitch_count increases BB/EV; illegal sub prevented by rule guard.

12) Manager/AI Strategy (Required)
- Lineups, substitutions, bullpen, tactics
  - Acceptance Criteria:
    - Heuristics for pinch-hitting/pinch-running, defensive replacements, and bullpen selection based on leverage and platoon.
    - Bunt/squeeze and IBB decisions with tunable aggressiveness; logged in play-by-play.
    - Tests: high leverage triggers better reliever; L/R platoon advantage increases PH likelihood.

13) Weather & Park Dynamics (Required)
- In-game changes and effects
  - Acceptance Criteria:
    - Weather can evolve across innings (temp, wind); roof open/close toggles by park.
    - Weather feeds BIP carry multipliers and grip (affecting control) within calibrated bounds.
    - Tests: higher temp slightly raises HR%; strong headwind reduces HR and increases FB outs.

14) ETL & Training (Real Data)
- Build PA training table from Retrosheet
  - Acceptance Criteria:
    - Script `scripts/build_pa_table.py` reads `data/processed/retrosheet_events.csv` and writes `artifacts/pa_table.parquet` with one row/PA including: batter_id, pitcher_id, base/outs, count, inning/half, park_id/date, outcome class {BB,K,HBP,IP_OUT,1B,2B,3B,HR}.
    - README snippet describing schema and required columns.

- Train PA model on real data
  - Acceptance Criteria:
    - `scripts/train_pa.py` loads `artifacts/pa_table.parquet`, builds features (ratings + context), splits train/val, trains multinomial model, saves `artifacts/pa_model.joblib`.
    - Prints class frequencies and validation log loss; exits non-error.
    - Server loads the artifact automatically and changes outputs vs baseline on a smoke request.

15) Calibration (Held-out)
- Calibrate with held-out set
  - Acceptance Criteria:
    - `scripts/calibrate_pa.py` consumes model + held-out set, fits `MultiClassPlattCalibrator`, saves `artifacts/pa_calibrator.joblib`.
    - Outputs reliability summary (mean raw -> mean cal | true rate) without errors.
    - A simple test verifies calibrated probabilities still sum to 1 and differ from raw for at least one class.

16) Evaluation & Reporting
- Reliability and realism checks
  - Acceptance Criteria:
    - Script `scripts/report_eval.py` computes overall and slice metrics (log loss, Brier, reliability bins) and prints/plots summary.
    - Simple RE table comparison: simulated vs historical RE24 differences summarized; stored as `artifacts/eval_report.txt`.

17) ONNX Export & Runtime (Performance)
- Export trained model to ONNX and enable runtime switch
  - Acceptance Criteria:
    - `scripts/export_onnx.py` writes `artifacts/pa_model.onnx`.
    - Add `src/baseball_sim/serve/onnx_runtime.py` to load and infer; config switch `model.use_onnx=true` routes inference via ONNX.
    - Parity check: ONNX vs joblib outputs match within tolerance (e.g., L1 diff < 1e-3 on a sample batch).

18) Integration Enhancements
- Players cache improvements
  - Acceptance Criteria:
    - Players CSV loader tolerates missing/extra columns and logs counts; memoized cache built at startup with a reload endpoint `/admin/reload-players` (protected by simple token env var).

19) Data Sources & Licensing
- Data attribution and source registry
  - Acceptance Criteria:
    - Add `DATA_SOURCES.md` with Retrosheet’s required notice and links to Statcast/MLB StatsAPI terms.
    - Footer/console notice printed on server start referencing data sources if configured.

20) Season/League Simulation (Optional)
- Schedule, standings, playoffs, rosters
  - Acceptance Criteria:
    - Load schedule, simulate games, update standings; simple roster rules for fatigue and rest days.
    - Tests: standings reflect wins/losses; rotation advances; fatigue recovers with rest.

21) Injury Model & Recovery (Optional)
- Pre-game and in-game injuries
  - Acceptance Criteria:
    - Probabilistic injuries by role and fatigue; durations; replace in lineup; recovery over days.
    - Tests: higher fatigue -> higher injury risk; injuries remove players correctly and insert subs.
