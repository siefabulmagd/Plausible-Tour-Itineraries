# P6 - Plausible Tour Itineraries (Neural network + Search + CSP)

City: **Melbourne** (60-place pool). Data: Flickr tour dataset, Lim et al. (`computationalmedia/tour-cikm16`).
All numbers below were produced by `run_all.py` (see `results/run.log`); nothing is hand-edited.

## Run
```bash
git clone https://github.com/computationalmedia/tour-cikm16.git   # the data is NOT redistributed here
pip install numpy pandas matplotlib
python -m unittest discover -s tests -v    # 19 tests: CSP, search vs brute force, NN gradient check
python run_all.py                           # ~3 min, writes results/
```

## Architecture
| file | role |
|---|---|
| `p6/data.py` | load, sort chronologically, pool selection, segments, travel-time model, user-level folds |
| `p6/models.py` | Markov-count and popularity baselines, **two-head NN** (shared trunk -> next-place head, log-duration head) |
| `p6/csp.py` | category-pattern CSP: `bt`, `fc`, `ac3` solvers |
| `p6/search.py` | UCS, A* (relaxed-problem heuristic), greedy, brute force |
| `p6/pipeline.py` | CSP -> search -> durations/budget -> best plan, with fallbacks |
| `p6/experiments.py` | folds, F1, Experiments 1-3, brute-force verification |

## Assumptions (not in the dataset - change them in `p6/config.py`)
* **Timezone**: raw timestamps look shifted (hour histogram peaks at 02-05h). I add +10 h. Unverified.
* **Travel time**: generated - faster of walking (4.5 km/h) and transit (10 min wait + 20 km/h), x1.3 detour, 5-min ticks.
* **Budgets and must-see category** are generated from each real trajectory: budget = max(1.5 x real elapsed time rounded to 30 min, 60 min x stops); must-see = category of a random interior real stop.
* **Durations**: visits < 5 min (photo-timestamp noise, 55-64% of rows) are *not used* as duration labels (the brief says "clip"; flag `min_dur_s`). The duration head therefore predicts the length of a *genuine* visit (median ~45 min).
* Pool chosen from all-data visit counts (>=20 visits, top 60). Candidate-set choice only; no transition labels used.
* The NN is **NumPy with hand-written back-propagation** (verified by finite differences) because PyTorch could not be installed in the build sandbox.

## Results (5-fold, user-level cross-validation; test users only)
**Next place** (1,929 test transitions; NLL lower is better)

| model | NLL | top-1 | top-3 |
|---|---|---|---|
| uniform | 4.078 | 0.017 | 0.051 |
| popularity | 3.830 | 0.076 | 0.162 |
| Markov counts | 3.565 | 0.142 | 0.297 |
| NN without hour | 3.510 | 0.128 | 0.295 |
| **NN (place + hour)** | 3.579 | 0.114 | 0.261 |

NN - Markov = +0.004 NLL [-0.007, +0.015]: **no evidence the NN beats the count baseline**. Adding the hour makes it worse (+0.017 [+0.011, +0.024] vs no-hour).

**Duration** (log-seconds MAE): global mean 1.117, per-place mean 1.068, NN head 1.069 (median error ~34 min). The NN head is no better than per-place means.

**Search** (verified against brute force on 375 runs: UCS and A* matched in 375/375). Experiment 1: A* expands 17-50 nodes at 7-8 stops vs 3,500-12,200 for UCS (same optimal cost in 60/60); greedy is 5-9% above optimal and fails to finish in 8-25% of cases at some lengths.

**CSP** (Experiment 2): BT, FC and AC-3 return identical solution sets (0 disagreements). FC cuts nodes ~2x over BT; **AC-3 adds nothing beyond FC** (same nodes, more checks) because the only binary constraint is "!="; on infeasible budgets FC/AC-3 prove infeasibility in 1 node vs ~121k for BT at 8 stops. On easy "first 10 patterns" BT is actually fastest in wall-clock (inference overhead).

**Experiment 3** (F1 against real held-out trajectories, interior stops, 211 queries, 95% user-bootstrap CI): random 0.103 [0.08, 0.13]; popularity 0.285; Markov 0.294; NN 0.286 [0.25, 0.33]; NN-no-hour 0.321. All learned costs beat random by ~0.18; differences among them are small and mostly not significant.

## Limitations
* ~1,300 training transitions over 60 places is little data; this limits the NN.
* F1 is dominated by popularity and the CSP's category structure; it barely separates the cost models.
* Real trajectories are short (only 211 segments with >=4 stops); 7-8 stop test cases number ~25.
* Durations are lower bounds from photo times; the timezone offset and travel model are assumptions.
