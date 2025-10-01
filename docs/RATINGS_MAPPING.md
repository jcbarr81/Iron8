# Ratings → Feature Mapping (Authoritative)

All ratings are on 0–99. We use a consistent transform:

```
z(x) = clamp((x - 50) / 15, -3, 3)
```

## Batters
- `gf` (GB/FB ratio): ↑gf ⇒ lower Launch Angle (LA), more GB, higher DP risk.
- `pl` (pull tendency): ↑pl ⇒ spray shifts toward pull side (by handedness).
- `sc` (RISP performance): Applied **only when RISP** (runner at 2B and/or 3B). Adds a **modest** bonus to hit logits, small penalty to K.
- `fa` (fielding, primary position): Defensive OAA-like conversion on relevant BIP types.
- `arm`: Outfield/IF arm—affects runner advancement and assists.

## Pitchers
- `arm` (or `as`): velocity proxy feeding “stuff”.
- Pitch types: `fb`, `sl`, `si`, `cu` (changeup), `cb` (curveball), `scb` (screwball), `kn` (knuckleball).
  - `fb/sl` push K% and air contact; `si/cb` push GB; `kn/scb` increase contact variance.
- `fa`: pitcher fielding—bunts, comebackers, PFP.
- Other (if present): `control` ↓BB; `movement` ↓HR/hard contact; `hold_runner` ↓SB; `endurance` shapes TTO/fatigue.

## Model Hooks
- PA Outcome head uses: `bat_contact/ch` (if present), `bat_power/ph` (if present), `pit_control`, `pit_move`, `pit_stuff` (from `arm`+mix), `RISP_gate(sc)`, and base/outs/count/park/handedness.
- BIP uses: `ph` (EV↑), `gf` (LA↓), `pl` (spray→pull), defense `fa`, arm, park rules.
