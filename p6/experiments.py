"""Training folds, query generation, F1 evaluation and the three experiments."""
import itertools
import math
import random
import time

import numpy as np
import pandas as pd

from . import csp
from .config import Config
from .data import fold_split
from .models import (MarkovCounts, Popularity, duration_baselines, eval_duration, eval_next, train_two_head,
                     tune_markov_alpha)
from .pipeline import Planner, Query
from .search import SearchProblem, astar, brute_force, greedy, ucs

KINDS = ["nn", "nn_nohour", "markov", "pop"]


# =============================================================================== folds
def build_fold(ds, fold_of, fold, cfg: Config, log=print):
    tr_u, va_u, te_u = fold_split(ds, fold_of, fold, cfg.val_frac, cfg.seed)
    T, V = ds.trans, ds.visits
    sel = lambda df, us: df[df.user.isin(us)]
    ttr, tva, tte = sel(T, tr_u), sel(T, va_u), sel(T, te_u)
    D = V[V.dur >= cfg.min_dur_s]
    dtr, dva, dte = sel(D, tr_u), sel(D, va_u), sel(D, te_u)
    alpha = tune_markov_alpha(ds.N, ttr, tva)
    markov = MarkovCounts(ds.N, alpha).fit(ttr.frm.values, ttr.to.values)
    pop = Popularity(ds.N).fit(sel(V, tr_u).poi.values)
    t0 = time.time()
    nn = train_two_head(ds.N, ds.C, ds.cat_of, ttr, tva, dtr, dva, cfg, seed=cfg.seed + fold, use_hour=True)
    nnh = train_two_head(ds.N, ds.C, ds.cat_of, ttr, tva, dtr, dva, cfg, seed=cfg.seed + fold, use_hour=False)
    log(f"  fold {fold}: train/val/test transitions {len(ttr)}/{len(tva)}/{len(tte)}, duration rows "
        f"{len(dtr)}/{len(dva)}/{len(dte)}, NN best epoch {nn['best_epoch']} ({time.time() - t0:.0f}s)")
    tables = {"nn": nn["nll"], "nn_nohour": nnh["nll"], "markov": markov.nll_table(), "pop": pop.nll_table()}

    # --- metrics (test users only) -------------------------------------------------------------
    nxt, per_trans = [], []
    uni = math.log(ds.N - 1)
    for k in KINDS:
        m = eval_next(tables[k], tte)
        nxt.append({"fold": fold, "model": k, **m})
        loss = tables[k][tte.hour_dep.values, tte.frm.values, tte.to.values]
        per_trans.append(pd.DataFrame({"fold": fold, "model": k, "user": tte.user.values, "loss": loss}))
    nxt.append({"fold": fold, "model": "uniform", "nll": uni, "top1": 1 / (ds.N - 1), "top3": 3 / (ds.N - 1), "n": len(tte)})
    glob, perp = duration_baselines(dtr, ds.N)
    dur = [{"fold": fold, "model": "global mean", **eval_duration(glob, dte)},
           {"fold": fold, "model": "per-place mean", **eval_duration(perp, dte)},
           {"fold": fold, "model": "NN duration head", **eval_duration(nn["dur_min"], dte)}]

    # category transition prior for CSP pattern ranking (train transitions only)
    cc = np.ones((ds.C, ds.C))
    np.add.at(cc, (ds.cat_of[ttr.frm.values], ds.cat_of[ttr.to.values]), 1)
    np.fill_diagonal(cc, 0)
    big = np.log(np.where(cc > 0, cc / cc.sum(1, keepdims=True), 1e-12))
    planner = Planner(ds, tables, nn["dur_min"], big, cfg.lam, cfg.max_expansions)
    rng = np.random.default_rng(cfg.seed + 7 * fold)
    return {"fold": fold, "planner": planner, "tables": tables, "dur_min": nn["dur_min"], "history": nn["history"],
            "best_epoch": nn["best_epoch"], "alpha": alpha, "next": nxt, "dur": dur,
            "per_trans": pd.concat(per_trans), "val_queries": make_queries(ds, va_u, rng),
            "test_queries": make_queries(ds, te_u, rng), "users": (tr_u, va_u, te_u)}


# =============================================================================== queries & F1
def ceil_to(x, m):
    return int(math.ceil(x / m) * m)


def make_queries(ds, users, rng, max_n=8):
    """One query per real trajectory segment with >= 4 stops (first `max_n` stops).
    GENERATED parts: the budget (1.5 x real elapsed time, rounded up to 30 min, at least 60 min per stop)
    and the must-see category (category of a random interior stop of the real trajectory)."""
    segs = ds.segments[ds.segments.user.isin(users) & (ds.segments.n >= 4)]
    vis = {s: g for s, g in ds.visits[ds.visits.seg.isin(segs.seg)].groupby("seg")}
    out = []
    for r in segs.itertuples():
        g = vis[r.seg].iloc[:max_n]
        pois = [int(p) for p in g.poi]
        n = len(pois)
        if pois[0] == pois[-1]:
            continue
        span = (g.end.max() - g.start.min()) / 60.0
        budget = max(ceil_to(1.5 * span, 30), 60 * n)
        must = int(ds.cat_of[pois[int(rng.integers(1, n - 1))]])
        out.append({"seg": int(r.seg), "user": r.user, "real": pois, "span": span,
                    "q": Query(pois[0], pois[-1], n, budget, must, int(g.tmin.iloc[0] // 5 * 5))})
    return out


def set_f1(pred, real):
    p, r = set(pred), set(real)
    tp = len(p & r)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(r)
    return 2 * pr * rc / (pr + rc)


def pairs_f1(pred, real):
    pairs = lambda x: {(x[i], x[j]) for i in range(len(x)) for j in range(i + 1, len(x))}
    return set_f1(list(pairs(pred)), list(pairs(real)))


def score(pred, real):
    if not pred:
        return {"f1_all": 0.0, "f1_int": 0.0, "pairs_f1": 0.0}
    return {"f1_all": set_f1(pred, real), "f1_int": set_f1(pred[1:-1], real[1:-1]),
            "pairs_f1": pairs_f1(pred, real)}


def eval_plans(fold, queries, kinds, lam_by_kind, with_random=True, seed=0):
    pl = fold["planner"]
    rows, rng = [], np.random.default_rng(seed + fold["fold"])
    for qd in queries:
        q, real = qd["q"], qd["real"]
        for k in kinds:
            pl.lam = lam_by_kind[k]
            t0 = time.perf_counter()
            plan = pl.plan(q, cost=k, algo="astar")
            dt = time.perf_counter() - t0
            rows.append({"fold": fold["fold"], "seg": qd["seg"], "user": qd["user"], "n": q.n, "kind": k,
                         "status": plan.status, "total_time": plan.total_time, "budget": q.budget,
                         "expansions": plan.expansions, "runtime_s": dt,
                         **score(plan.places(), real)})
        if with_random:
            path = pl.random_plan(q, rng)
            rows.append({"fold": fold["fold"], "seg": qd["seg"], "user": qd["user"], "n": q.n, "kind": "random",
                         "status": "ok" if path else "none", "total_time": None, "budget": q.budget,
                         "expansions": 0, "runtime_s": 0.0, **score(path or [], real)})
    return pd.DataFrame(rows)


# =============================================================================== lambda tuning
def tune_lambda(folds, grid=(0.0, 0.02, 0.05, 0.1, 0.25), kinds=KINDS, cap_per_fold=40, seed=0):
    """Pick the minutes->nats exchange rate separately for every cost model on VALIDATION queries."""
    rows = []
    for lam in grid:
        for f in folds:
            qs = f["val_queries"][:cap_per_fold]
            df = eval_plans(f, qs, kinds, {k: lam for k in kinds}, with_random=False, seed=seed)
            df["lam"] = lam
            rows.append(df)
    d = pd.concat(rows)
    tab = d.groupby(["kind", "lam"]).f1_int.mean().unstack()
    best = {k: float(tab.loc[k].idxmax()) for k in kinds}
    return best, tab


# =============================================================================== Experiment 1
def run_exp1(folds, per_n=12, seed=0, cap=200_000):
    rng = random.Random(seed)
    pool = {n: [] for n in range(4, 9)}
    for fi, f in enumerate(folds):
        for qd in f["test_queries"]:
            pool[qd["q"].n].append((fi, qd))
    rows = []
    for n in range(4, 9):
        items = pool[n]
        rng.shuffle(items)
        for fi, qd in items[:per_n]:
            pl = folds[fi]["planner"]
            q = qd["q"]
            pats, _, _ = pl.csp_patterns(q, "query", 1)
            if not pats:
                continue
            P = pl.problem(q, pats[0], "nn")
            res = {"ucs": ucs(P, cap), "astar": astar(P, cap), "greedy": greedy(P)}
            ref = res["astar"].cost if res["astar"].status == "ok" else None
            for a, r in res.items():
                rows.append({"n": n, "fold": fi, "seg": qd["seg"], "algo": a, "status": r.status, "cost": r.cost,
                             "total_time": r.total_time, "expansions": r.expansions, "generated": r.generated,
                             "runtime_s": r.runtime_s, "opt_cost": ref})
    return pd.DataFrame(rows)


# =============================================================================== Experiment 2
def lower_bound_time(spec):
    return (spec.dur_first + spec.dur_last + (spec.n - 2) * min(spec.dur.values()) + (spec.n - 1) * spec.hop)


def run_exp2(folds, per_n=15, seed=0, node_limit=300_000, repeats=3):
    rng = random.Random(seed)
    pool = {n: [] for n in range(4, 9)}
    for fi, f in enumerate(folds):
        for qd in f["test_queries"]:
            pool[qd["q"].n].append((fi, qd))
    rows, mismatches = [], 0
    for n in range(4, 9):
        items = pool[n]
        rng.shuffle(items)
        for fi, qd in items[:per_n]:
            pl = folds[fi]["planner"]
            spec = pl.spec(qd["q"])
            tight = csp.CSPSpec(**{**spec.__dict__, "budget": 0.9 * lower_bound_time(spec)})
            scen = [("first10", spec, dict(max_solutions=10)), ("unsat", tight, {})]
            if n <= 6:
                scen.append(("all", spec, {}))
            for name, sp, kw in scen:
                sols_by = {}
                for m in ("bt", "fc", "ac3"):
                    best_ms, st = None, None
                    for _ in range(repeats):
                        sols, st = csp.solve(sp, m, node_limit=node_limit, **kw)
                        best_ms = st.time_ms if best_ms is None else min(best_ms, st.time_ms)
                    sols_by[m] = (set(sols), st.truncated)
                    rows.append({"n": n, "scenario": name, "method": m, "nodes": st.nodes, "backtracks": st.backtracks,
                                 "checks": st.checks, "time_ms": best_ms, "solutions": st.solutions,
                                 "truncated": st.truncated, "fold": fi, "seg": qd["seg"]})
                if name in ("all", "unsat") and not any(t for _, t in sols_by.values()):
                    if not (sols_by["bt"][0] == sols_by["fc"][0] == sols_by["ac3"][0]):
                        mismatches += 1
    return pd.DataFrame(rows), mismatches


# =============================================================================== search verification
def verify_bruteforce(fold, trials=40, seed=0):
    """UCS / A* on REAL model costs, restricted to a 12-place sub-pool, against exhaustive enumeration."""
    pl, ds = fold["planner"], fold["planner"].ds
    rnd = random.Random(seed)
    nll_full = fold["tables"]["nn"]
    dur_full = np.array(pl.dur5)
    tt_full = np.array(pl.tt5)
    big = [c for c in range(ds.C) if (ds.cat_of == c).sum() >= 4]
    rows = []
    for t in range(trials):
        cats = rnd.sample(big, 3)
        S = []
        for c in cats:
            S += rnd.sample([int(p) for p in np.where(ds.cat_of == c)[0]], 4)
        cat_sub = [i // 4 for i in range(12)]
        nll = nll_full[np.ix_(range(24), S, S)].tolist()
        dur5 = dur_full[:, S].tolist()
        tt5 = tt_full[np.ix_(S, S)].tolist()
        n = rnd.choice([4, 5, 6])
        start = rnd.randrange(12)
        end = rnd.choice([i for i in range(12) if cat_sub[i] != cat_sub[start]])
        pat = [cat_sub[start]]
        for _ in range(1, n - 1):
            pat.append(rnd.choice([c for c in range(3) if c != pat[-1]]))
        if pat[-1] == cat_sub[end]:
            pat[-2:] = [pat[-2], [c for c in range(3) if c not in (pat[-2], cat_sub[end])][0]] if n > 3 else pat[-2:]
        pat.append(cat_sub[end])
        T0 = rnd.choice([540, 600, 780])
        P0 = SearchProblem(pat, start, end, cat_sub, nll, dur5, tt5, T0, 0.05, None)
        ref0 = brute_force(P0)
        if ref0.status != "ok":
            continue
        for bname, B in (("none", None), ("loose", int(ref0.total_time * 1.25)), ("tight", int(ref0.total_time * 0.9))):
            P = SearchProblem(pat, start, end, cat_sub, nll, dur5, tt5, T0, 0.05, B)
            bf, u, a, g = brute_force(P), ucs(P), astar(P), greedy(P)
            ok = bf.status == "ok"
            rows.append({"trial": t, "n": n, "budget": bname, "feasible": ok, "bf_cost": bf.cost if ok else None,
                         "ucs_match": (u.status == "ok" and abs(u.cost - bf.cost) < 1e-9) if ok else u.status != "ok",
                         "astar_match": (a.status == "ok" and abs(a.cost - bf.cost) < 1e-9) if ok else a.status != "ok",
                         "greedy_cost": g.cost if g.status == "ok" else None,
                         "greedy_optimal": (g.status == "ok" and abs(g.cost - bf.cost) < 1e-9) if ok else None,
                         "bf_generated": bf.generated, "ucs_exp": u.expansions, "astar_exp": a.expansions})
    return pd.DataFrame(rows)


# =============================================================================== statistics
def cluster_boot(df, value_cols, n_boot=2000, seed=0):
    """Mean of each column with a 95% CI obtained by resampling USERS (queries of one user are correlated)."""
    rng = np.random.default_rng(seed)
    g = df.groupby("user")[value_cols].agg(["sum", "count"])
    users = g.index.values
    out = {}
    for c in value_cols:
        s, k = g[(c, "sum")].values, g[(c, "count")].values
        idx = rng.integers(0, len(users), (n_boot, len(users)))
        means = s[idx].sum(1) / k[idx].sum(1)
        out[c] = (s.sum() / k.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5))
    return out
