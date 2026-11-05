"""End-to-end planner:  CSP patterns -> search with model costs -> duration/budget check -> best plan.

    1. CSP produces up to 10 valid category patterns (ranked by category-transition plausibility)
    2. Search instantiates every pattern with real places using the chosen transition-cost table
    3. Predicted visit durations + travel times give arrival times and the total time
    4. Plans over budget are rejected (the search never returns one)
    5. The most plausible valid plan is returned
    Fallbacks: if nothing fits, report the smallest budget that works and return the cheapest plan
    (lowest model cost, budget ignored) together with a warning.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import csp
from .search import SearchProblem, astar, ucs, greedy


@dataclass
class Query:
    start: int                 # place index
    end: int
    n: int                     # number of stops including start and end
    budget: int               # minutes
    must: int                  # must-see category index
    T0: int = 540              # clock minute of day when the tour starts


@dataclass
class Plan:
    status: str                # ok | over_budget | infeasible
    stops: list = field(default_factory=list)
    cost: float = None
    total_time: int = None
    min_budget: int = None     # smallest budget that works (only reported when status != ok)
    warning: str = ""
    patterns_valid: int = 0
    patterns_tried: int = 0
    expansions: int = 0
    pattern: tuple = ()

    def places(self):
        return [s["place"] for s in self.stops]


def clock(T0, e):
    m = (T0 + e) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


class Planner:
    def __init__(self, ds, nll_tables: dict, dur_min: np.ndarray, cat_logbigram: np.ndarray, lam: float,
                 max_expansions=200_000):
        self.ds, self.lam, self.max_exp = ds, lam, max_expansions
        self.N, self.C = ds.N, ds.C
        self.cat_of = [int(c) for c in ds.cat_of]
        self.tt5 = ds.tt.astype(int).tolist()
        self.dur_min = dur_min
        self.dur5 = (5 * np.ceil(dur_min / 5.0)).astype(int).tolist()
        self.nll = {k: v.tolist() for k, v in nll_tables.items()}
        self.bigram = cat_logbigram
        sizes = np.bincount(ds.cat_of, minlength=ds.C)
        self.cap = {ds.cats[c]: int(min(2, sizes[c])) for c in range(ds.C)}
        dur_avg = dur_min[9:18].mean(0)                         # expected minutes per place, daytime
        self.dur_avg = dur_avg
        self.cat_dur = {ds.cats[c]: float(dur_avg[ds.cat_of == c].mean()) for c in range(ds.C)}
        off = ds.tt[~np.eye(ds.N, dtype=bool)]
        self.hop = float(np.percentile(off, 25))                # optimistic-ish expected hop time

    # ------------------------------------------------------------------ CSP side
    def spec(self, q: Query, budget="query") -> csp.CSPSpec:
        ds = self.ds
        return csp.CSPSpec(
            n=q.n, cats=list(ds.cats), first=ds.cats[self.cat_of[q.start]], last=ds.cats[self.cat_of[q.end]],
            must=ds.cats[q.must], cap=dict(self.cap), dur=dict(self.cat_dur),
            dur_first=float(self.dur_avg[q.start]), dur_last=float(self.dur_avg[q.end]), hop=self.hop,
            budget=q.budget if budget == "query" else budget)

    def pattern_score(self, pat):
        ci = [self.ds.cats.index(c) for c in pat]
        return float(sum(self.bigram[a, b] for a, b in zip(ci[:-1], ci[1:])))

    def csp_patterns(self, q, budget="query", k=10, max_enum=2000, method="ac3", by="plaus"):
        spec = self.spec(q, budget)
        cats = self.ds.cats
        if by == "plaus":
            def order(i, prev, dom):
                if prev is None:
                    return sorted(dom)
                p = cats.index(prev)
                return sorted(dom, key=lambda c: -self.bigram[p, cats.index(c)])
        else:  # shortest expected duration first
            def order(i, prev, dom):
                return sorted(dom, key=lambda c: self.cat_dur[c])
        sols, st = csp.solve(spec, method, max_solutions=max_enum, order_fn=order)
        if by == "plaus":
            sols.sort(key=lambda p: -self.pattern_score(p))
        else:
            sols.sort(key=lambda p: spec.total_time(p))
        return sols[:k], len(sols), st

    # ------------------------------------------------------------------ search side
    def problem(self, q: Query, pattern, cost="nn", budget="query", mode="plaus", lam=None):
        pidx = [self.ds.cats.index(c) for c in pattern]
        return SearchProblem(pidx, q.start, q.end, self.cat_of, self.nll[cost], self.dur5, self.tt5, q.T0,
                             self.lam if lam is None else lam, q.budget if budget == "query" else budget, mode)

    def run_search(self, P, algo):
        return {"astar": astar, "ucs": ucs}[algo](P, self.max_exp) if algo != "greedy" else greedy(P)

    def _to_plan(self, q, res, pattern, status, warning, **kw):
        ds = self.ds
        stops = []
        for i, p in enumerate(res.path):
            stops.append({"place": p, "poi_id": int(ds.pois.poiID[p]), "category": ds.cats[self.cat_of[p]],
                          "arrive": clock(q.T0, res.arrivals[i]), "stay_min": res.durs[i],
                          "leave": clock(q.T0, res.arrivals[i] + res.durs[i]),
                          "travel_from_prev_min": 0 if i == 0 else self.tt5[res.path[i - 1]][p]})
        return Plan(status=status, stops=stops, cost=res.cost, total_time=res.total_time, warning=warning,
                    pattern=tuple(pattern), **kw)

    # ------------------------------------------------------------------ main entry
    def plan(self, q: Query, cost="nn", algo="astar", n_patterns=10, max_enum=2000) -> Plan:
        pats, n_valid, _ = self.csp_patterns(q, "query", n_patterns, max_enum)
        exp, best, tried = 0, None, 0
        for pat in pats:
            P = self.problem(q, pat, cost)
            r = self.run_search(P, algo)
            exp += r.expansions
            tried += 1
            if r.status == "ok" and (best is None or r.cost < best[0].cost):
                best = (r, pat)
        if best is not None:
            return self._to_plan(q, best[0], best[1], "ok", "", patterns_valid=n_valid, patterns_tried=tried,
                                 expansions=exp)

        # ---- fallback: nothing fits the budget ------------------------------------------------
        pats_nb, n_nb, _ = self.csp_patterns(q, None, n_patterns, max_enum)
        if not pats_nb:
            return Plan("infeasible", warning="No category pattern satisfies the structural constraints "
                        "(neighbours differ, must-see present, <=2 per category) for these endpoints.",
                        patterns_valid=0)
        # smallest budget that works: time-optimal plan over the shortest patterns
        short, _, _ = self.csp_patterns(q, None, n_patterns, max_enum=3000, by="time")
        mins = []
        for pat in short:
            r = astar(self.problem(q, pat, cost, budget=None, mode="time"), self.max_exp)
            exp += r.expansions
            if r.status == "ok":
                mins.append(r.total_time)
        min_budget = min(mins) if mins else None
        # cheapest (most plausible) plan ignoring the budget
        cheap = None
        for pat in pats_nb:
            r = self.run_search(self.problem(q, pat, cost, budget=None), algo)
            exp += r.expansions
            if r.status == "ok" and (cheap is None or r.cost < cheap[0].cost):
                cheap = (r, pat)
        if cheap is None:
            return Plan("infeasible", min_budget=min_budget, warning="Search found no plan.", expansions=exp)
        over = cheap[0].total_time - q.budget
        warn = (f"WARNING: no itinerary fits {q.budget} min. Smallest budget that works: "
                f"{min_budget} min. Returning the most plausible plan, which needs {cheap[0].total_time} min "
                f"({over} min over budget).")
        return self._to_plan(q, cheap[0], cheap[1], "over_budget", warn, min_budget=min_budget,
                             patterns_valid=0, patterns_tried=tried, expansions=exp)

    # ------------------------------------------------------------------ non-model reference
    def random_plan(self, q: Query, rng, tries=200):
        """Reference: best-plausibility pattern, then uniformly random distinct places per position."""
        pats, _, _ = self.csp_patterns(q, "query", 1)
        if not pats:
            pats, _, _ = self.csp_patterns(q, None, 1)
        if not pats:
            return None
        pat = pats[0]
        P = self.problem(q, pat, "nn", budget="query")
        from .search import evaluate_path
        for _ in range(tries):
            path = [q.start]
            for pos in range(1, q.n - 1):
                opts = [p for p in P.cands[pos] if p not in path]
                if not opts:
                    break
                path.append(int(rng.choice(opts)))
            else:
                path.append(q.end)
                _, _, _, total = evaluate_path(P, path)
                if total <= q.budget:
                    return path
        return path if len(path) == q.n else None
