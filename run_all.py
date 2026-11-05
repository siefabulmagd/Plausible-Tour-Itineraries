"""Run the complete P6 study:  python run_all.py [--quick]
Writes tables (CSV/JSON) and figures to results/.  Every number in the report comes from this script."""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p6.config import Config
from p6.data import load_dataset, user_folds
from p6.experiments import (KINDS, build_fold, cluster_boot, eval_plans, run_exp1, run_exp2, tune_lambda,
                            verify_bruteforce)

QUICK = "--quick" in sys.argv
OUT = "results_quick" if QUICK else "results"
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/run.log", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


cfg = Config()
ds = load_dataset(cfg)
fold_of = user_folds(ds, cfg.k_folds, cfg.seed)
K = 2 if QUICK else cfg.k_folds
log(f"City {cfg.city}: pool {ds.N} places, {ds.C} categories, {len(ds.trans)} transitions, "
    f"{(ds.segments.n >= 4).sum()} real segments with >=4 stops, users {ds.visits.user.nunique()}")
log("Training folds ...")
folds = [build_fold(ds, fold_of, k, cfg, log) for k in range(K)]
S = {"config": cfg.__dict__, "n_places": ds.N, "categories": ds.cats}

# ------------------------------------------------------------------ deliverable: NN vs Markov
nx = pd.DataFrame([r for f in folds for r in f["next"]])
nxt = []
for m, g in nx.groupby("model"):
    w = g.n / g.n.sum()
    nxt.append({"model": m, "nll": (g.nll * w).sum(), "nll_sd_over_folds": g.nll.std(), "top1": (g.top1 * w).sum(),
                "top3": (g.top3 * w).sum(), "n_test": int(g.n.sum())})
nxt = pd.DataFrame(nxt).set_index("model").loc[["uniform", "pop", "markov", "nn_nohour", "nn"]]
pt = pd.concat([f["per_trans"] for f in folds])
w = pt.pivot_table(index=["fold", "user"], columns="model", values="loss", aggfunc="sum")
cnt = pt.groupby(["fold", "user"]).size()
rng = np.random.default_rng(0)
dd = {}
for a, b in [("nn", "markov"), ("nn", "nn_nohour"), ("nn", "pop"), ("markov", "pop")]:
    s = (w[a] - w[b]).values; c = cnt.loc[w.index].values
    idx = rng.integers(0, len(s), (2000, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    dd[f"{a} - {b}"] = (s.sum() / c.sum(), *np.percentile(bs, [2.5, 97.5]))
nxt.to_csv(f"{OUT}/next_place_metrics.csv")
log("\n=== NEXT-PLACE (test users, pooled over folds; lower NLL is better) ===")
log(nxt.round(3).to_string())
log("NLL differences (negative = first model better), 95% user-bootstrap CI:")
for k, v in dd.items():
    log(f"  {k:22s} {v[0]:+.3f}  [{v[1]:+.3f}, {v[2]:+.3f}]")
S["next_place"] = nxt.round(4).reset_index().to_dict("records")
S["nll_diffs"] = {k: [round(x, 4) for x in v] for k, v in dd.items()}

du = pd.DataFrame([r for f in folds for r in f["dur"]])
dur = []
for m, g in du.groupby("model"):
    w_ = g.n / g.n.sum()
    dur.append({"model": m, "mae_log": (g.mae_log * w_).sum(), "rmse_log": (g.rmse_log * w_).sum(),
                "median_abs_err_min": (g.medAE_min * w_).sum(), "n_test": int(g.n.sum())})
dur = pd.DataFrame(dur).set_index("model").loc[["global mean", "per-place mean", "NN duration head"]]
dur.to_csv(f"{OUT}/duration_metrics.csv")
log("\n=== VISIT DURATION (test visits >= 5 min; error in log-seconds) ===")
log(dur.round(3).to_string())
S["duration"] = dur.round(4).reset_index().to_dict("records")

# ------------------------------------------------------------------ search verification vs brute force
log("\n=== SEARCH vs BRUTE FORCE on real NN costs (12-place sub-pools) ===")
vf = pd.concat([verify_bruteforce(f, trials=10 if QUICK else 25, seed=f["fold"]) for f in folds])
feas = vf[vf.feasible]
log(f"runs {len(vf)} (feasible {len(feas)}): UCS == brute force in {int(vf.ucs_match.sum())}/{len(vf)}, "
    f"A* == brute force in {int(vf.astar_match.sum())}/{len(vf)}; greedy optimal in "
    f"{int(feas.greedy_optimal.sum())}/{len(feas)} feasible runs")
log(f"mean states: brute force {feas.bf_generated.mean():.0f}, UCS expansions {feas.ucs_exp.mean():.0f}, "
    f"A* expansions {feas.astar_exp.mean():.0f}")
vf.to_csv(f"{OUT}/verify_bruteforce.csv", index=False)
S["bruteforce"] = {"runs": len(vf), "feasible": int(len(feas)), "ucs_match": int(vf.ucs_match.sum()),
                   "astar_match": int(vf.astar_match.sum()), "greedy_optimal": int(feas.greedy_optimal.sum())}

# ------------------------------------------------------------------ lambda tuning (validation queries)
log("\n=== Tuning lambda (nats per travel minute) on VALIDATION queries ===")
lam_best, lam_tab = tune_lambda(folds, grid=(0.0, 0.05, 0.25) if QUICK else (0.0, 0.02, 0.05, 0.1, 0.25),
                                cap_per_fold=15 if QUICK else 40)
log(lam_tab.round(3).to_string()); log("chosen:", lam_best)
lam_tab.to_csv(f"{OUT}/lambda_tuning_val.csv")
S["lambda"] = lam_best

# ------------------------------------------------------------------ Experiment 3 (+ itinerary quality)
log("\n=== EXPERIMENT 3: transition costs, F1 vs real held-out trajectories ===")
parts = [eval_plans(f, f["test_queries"], KINDS, lam_best, with_random=True, seed=1) for f in folds]
e3 = pd.concat(parts)
e3.to_csv(f"{OUT}/exp3_per_query.csv", index=False)
order = ["random", "pop", "markov", "nn_nohour", "nn"]
rows = []
for k in order:
    g = e3[e3.kind == k]
    ci = cluster_boot(g, ["f1_all", "f1_int", "pairs_f1"])
    rows.append({"cost model": k, "queries": len(g),
                 **{f"{c} mean": ci[c][0] for c in ci}, **{f"{c} lo": ci[c][1] for c in ci}, **{f"{c} hi": ci[c][2] for c in ci},
                 "ok%": 100 * (g.status == "ok").mean(), "over_budget%": 100 * (g.status == "over_budget").mean(),
                 "infeasible%": 100 * g.status.isin(["infeasible", "none"]).mean()})
e3s = pd.DataFrame(rows).set_index("cost model")
e3s.to_csv(f"{OUT}/exp3_summary.csv")
log(e3s[["queries", "f1_all mean", "f1_int mean", "f1_int lo", "f1_int hi", "pairs_f1 mean", "ok%", "over_budget%"]].round(3).to_string())
wide = e3.pivot_table(index=["fold", "seg", "user"], columns="kind", values=["f1_int", "pairs_f1"]).reset_index()
wide.columns = ["_".join(c).strip("_") for c in wide.columns]
pair = []
for met in ("f1_int", "pairs_f1"):
    for a, b in [("nn", "markov"), ("nn", "pop"), ("nn", "nn_nohour"), ("nn", "random"), ("markov", "pop")]:
        wide[f"d"] = wide[f"{met}_{a}"] - wide[f"{met}_{b}"]
        ci = cluster_boot(wide, ["d"])["d"]
        pair.append({"metric": met, "comparison": f"{a} - {b}", "diff": ci[0], "lo": ci[1], "hi": ci[2]})
pair = pd.DataFrame(pair)
pair.to_csv(f"{OUT}/exp3_paired_differences.csv", index=False)
log("Paired differences (positive = first better), 95% user-bootstrap CI:")
log(pair.round(3).to_string(index=False))
S["exp3"] = e3s.round(4).reset_index().to_dict("records"); S["exp3_paired"] = pair.round(4).to_dict("records")

# ------------------------------------------------------------------ Experiment 1
log("\n=== EXPERIMENT 1: UCS vs A* vs Greedy (NN costs, best pattern, 4-8 stops) ===")
for f in folds:
    f["planner"].lam = lam_best["nn"]
e1 = run_exp1(folds, per_n=4 if QUICK else 12)
e1.to_csv(f"{OUT}/exp1_search.csv", index=False)
e1["gap"] = np.where(e1.status.eq("ok") & e1.opt_cost.notna(), (e1.cost - e1.opt_cost) / e1.opt_cost.clip(lower=1e-9), np.nan)
agg = e1.groupby(["n", "algo"]).agg(queries=("seg", "count"), solved=("status", lambda s: (s == "ok").mean()),
                                    capped=("status", lambda s: (s == "capped").mean()),
                                    exp_mean=("expansions", "mean"), exp_median=("expansions", "median"),
                                    ms_median=("runtime_s", lambda x: 1000 * x.median()), gap_pct=("gap", lambda x: 100 * x.mean()))
agg.to_csv(f"{OUT}/exp1_summary.csv")
log(agg.round(2).to_string())
w1 = e1.pivot_table(index=["n", "fold", "seg"], columns="algo", values="cost")
both = w1.dropna(subset=["ucs", "astar"])
log(f"UCS cost == A* cost on {int((abs(both.ucs - both.astar) < 1e-9).sum())}/{len(both)} queries where both finished")
S["exp1"] = agg.round(3).reset_index().to_dict("records")

# ------------------------------------------------------------------ Experiment 2
log("\n=== EXPERIMENT 2: CSP  BT vs BT+FC vs BT+FC+AC-3 ===")
e2, mism = run_exp2(folds, per_n=4 if QUICK else 15)
e2.to_csv(f"{OUT}/exp2_csp.csv", index=False)
agg2 = e2.groupby(["scenario", "n", "method"]).agg(instances=("seg", "count"), nodes=("nodes", "mean"), backtracks=("backtracks", "mean"),
                                                    checks=("checks", "mean"), time_ms=("time_ms", "mean"),
                                                    valid_patterns=("solutions", "mean"), truncated=("truncated", "mean"))
agg2.to_csv(f"{OUT}/exp2_summary.csv")
log(agg2.round(1).to_string())
log(f"Solution-set disagreements between BT / FC / AC-3 (non-truncated runs): {mism}")
S["exp2"] = agg2.round(2).reset_index().to_dict("records"); S["exp2_mismatches"] = mism

# ------------------------------------------------------------------ example itinerary
f0 = folds[0]; pl = f0["planner"]
ex = next(q for q in f0["test_queries"] if q["q"].n == 5 and 540 <= q["q"].T0 <= 900)
pl.lam = lam_best["nn"]
plan = pl.plan(ex["q"], "nn")
log("\n=== EXAMPLE ===")
log(f"query: start={ds.pois.poiID[ex['q'].start]} ({ds.cats[ds.cat_of[ex['q'].start]]}), end={ds.pois.poiID[ex['q'].end]}, "
    f"{ex['q'].n} stops, budget {ex['q'].budget} min, must-see '{ds.cats[ex['q'].must]}', start {ex['q'].T0 // 60:02d}:{ex['q'].T0 % 60:02d}")
log(f"status {plan.status}; pattern {plan.pattern}; total {plan.total_time} min; real trajectory places {[int(ds.pois.poiID[p]) for p in ex['real']]}")
for s in plan.stops:
    log(f"  {s['arrive']}  {s['category']:<18} poi {s['poi_id']:<4} stay {s['stay_min']:>3} min  (travel {s['travel_from_prev_min']} min)")
json.dump({"query": {k: int(v) for k, v in ex["q"].__dict__.items()}, "plan": plan.__dict__}, open(f"{OUT}/example_plan.json", "w"), indent=1, default=str)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
col = {"random": "#9e9e9e", "pop": "#c9a227", "markov": "#4c78a8", "nn_nohour": "#8e6bbf", "nn": "#d1495b", "uniform": "#cccccc"}
# 1 next place
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ms = ["uniform", "pop", "markov", "nn_nohour", "nn"]
ax[0].bar(ms, nxt.loc[ms, "nll"], color=[col[m] for m in ms]); ax[0].set_title("Test NLL (lower = better)"); ax[0].set_ylim(3.2, 4.2)
ax[1].bar(ms, nxt.loc[ms, "top1"], color=[col[m] for m in ms]); ax[1].set_title("Top-1 accuracy")
ax[2].bar(ms, nxt.loc[ms, "top3"], color=[col[m] for m in ms]); ax[2].set_title("Top-3 accuracy")
for a in ax: a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_next_place.png"); plt.close()
# 2 training curves
h = pd.DataFrame(folds[0]["history"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(h.epoch, h.train_loss, label="train (next + duration)"); ax[0].plot(h.epoch, h.val_nll, label="val next-place NLL")
ax[0].axvline(folds[0]["best_epoch"], ls="--", c="grey"); ax[0].legend(); ax[0].set_xlabel("epoch"); ax[0].set_title("NN training (fold 0)")
ax[1].plot(h.epoch, h.val_dur_mse_std, c="#2a9d8f"); ax[1].axvline(folds[0]["best_epoch"], ls="--", c="grey")
ax[1].set_title("val duration MSE (standardised log)"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_training.png"); plt.close()
# 3 exp1
ac = {"ucs": "#4c78a8", "astar": "#d1495b", "greedy": "#9e9e9e"}
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
for a_, g in agg.reset_index().groupby("algo"):
    ax[0].plot(g.n, g.exp_mean, "o-", label=a_, c=ac[a_]); ax[1].plot(g.n, g.ms_median, "o-", label=a_, c=ac[a_])
    ax[2].plot(g.n, g.gap_pct, "o-", label=a_, c=ac[a_])
ax[0].set_yscale("log"); ax[0].set_title("Mean node expansions"); ax[1].set_yscale("log"); ax[1].set_title("Median run time (ms)")
ax[2].set_title("Mean cost gap to optimum (%)")
for a in ax: a.set_xlabel("stops"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_exp1_search.png"); plt.close()
# 4 exp2
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
mc = {"bt": "#9e9e9e", "fc": "#4c78a8", "ac3": "#d1495b"}
for i, sc in enumerate(["first10", "all", "unsat"]):
    g = agg2.reset_index(); g = g[g.scenario == sc]
    for m_, gg in g.groupby("method"):
        ax[i].plot(gg.n, gg.nodes, "o-", label={"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}[m_], c=mc[m_])
    ax[i].set_yscale("log"); ax[i].set_title({"first10": "first 10 patterns", "all": "all patterns (n<=6)", "unsat": "prove infeasible"}[sc]); ax[i].set_xlabel("stops")
    ax[i].set_ylabel("search nodes" if i == 0 else ""); ax[i].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_exp2_csp.png"); plt.close()
# 5 exp3
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, met, t in [(ax[0], "f1_int", "Set-F1, interior stops"), (ax[1], "pairs_f1", "Pairs-F1 (order)")]:
    m_ = e3s[f"{met} mean"]; lo = e3s[f"{met} lo"]; hi = e3s[f"{met} hi"]
    a.bar(order, m_, color=[col[o] for o in order], yerr=[m_ - lo, hi - m_], capsize=4); a.set_title(t); a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_exp3_f1.png"); plt.close()
# 6 example
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
ax[0].scatter(ds.pois.poiLon, ds.pois.poiLat, c="#dddddd", s=20)
xs = [ds.pois.poiLon[s["place"]] for s in plan.stops]; ys = [ds.pois.poiLat[s["place"]] for s in plan.stops]
ax[0].plot(xs, ys, "-o", c="#d1495b", label="generated")
rl = ex["real"]; ax[0].plot([ds.pois.poiLon[p] for p in rl], [ds.pois.poiLat[p] for p in rl], "--", c="#4c78a8", label="real tourist")
for i, s in enumerate(plan.stops): ax[0].annotate(str(i + 1), (xs[i], ys[i]), fontsize=9, xytext=(4, 4), textcoords="offset points")
ax[0].legend(); ax[0].set_title("Generated vs real route"); ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat")
for i, s in enumerate(plan.stops):
    a0 = ex["q"].T0 + plan.stops[i]["stay_min"] * 0
    h0 = int(s["arrive"][:2]) * 60 + int(s["arrive"][3:]); ax[1].barh(i, s["stay_min"], left=h0, color="#d1495b", alpha=.8)
    ax[1].text(h0 + 2, i, s["category"], va="center", fontsize=8, color="white")
ax[1].invert_yaxis(); ax[1].set_yticks(range(len(plan.stops))); ax[1].set_yticklabels([f"stop {i + 1}" for i in range(len(plan.stops))])
ax[1].set_xlabel("minute of day"); ax[1].set_title("Timeline (arrival, predicted stay)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_example.png"); plt.close()
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
log("\nDONE")
