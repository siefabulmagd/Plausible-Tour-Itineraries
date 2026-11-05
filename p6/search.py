"""Itinerary search.

Given a category pattern (from the CSP) the search picks one real place per position.

State   : (position in plan, current place, visited set, elapsed time)
Action  : go to an unvisited place whose category is the one required at the next position
Cost    : -log P_model(next | current, hour) + lam * travel_minutes          ("plaus" mode)
          (or elapsed minutes in "time" mode, used to find the smallest feasible budget)

Time model (all integers, minutes, multiples of the 5-minute tick so that the state is exactly
Markov and closed-set pruning is sound):
    arrive at u at elapsed a  ->  visit lasts d5[hour(a)][u]  ->  leave at dep = a + d
    hour used by the transition model = hour(dep)            (the tourist leaves at that hour)
    arrive at v at a_v = dep + tt5[u][v]
    total time = arrival at the last place + the visit duration there.

Algorithms
    ucs     uniform-cost search      f = g
    astar   A*                       f = g + h,   h = exact cost-to-go of a RELAXED problem
    greedy  myopic greedy            always takes the cheapest next edge (no backtracking)
    brute_force  exhaustive enumeration (used only to verify the others on small pools)

Heuristic.  h[pos][u] is the cheapest cost of completing the pattern from place u at position pos when
 (i) the visited-set constraint is dropped and (ii) every edge is charged its cheapest value over all
24 hours.  It is the exact shortest path in a relaxation, hence admissible, and it is consistent
(h(u) <= cmin(u,v) + h(v) <= c(u,v | any hour) + h(v)), so A* with a closed set returns optimal plans.
"""
import heapq
import itertools
import time
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class SearchProblem:
    pattern: list              # category index for every position (0..n-1)
    start: int
    end: int
    cat_of: list               # place -> category
    nll: list                  # nll[hour][u][v]  (python nested lists, +inf on diagonal)
    dur5: list                 # dur5[hour][u]   visit minutes (ints)
    tt5: list                  # tt5[u][v]       travel minutes (ints)
    T0: int = 540              # clock minute of day at which the itinerary starts
    lam: float = 0.05
    budget: int = None         # minutes; None = unconstrained
    mode: str = "plaus"        # 'plaus' or 'time'

    def __post_init__(self):
        self.n = len(self.pattern)
        self.N = len(self.cat_of)
        assert self.pattern[0] == self.cat_of[self.start] and self.pattern[-1] == self.cat_of[self.end]
        assert self.start != self.end
        self.cands = []
        for pos in range(self.n):
            if pos == 0:
                self.cands.append([self.start])
            elif pos == self.n - 1:
                self.cands.append([self.end])
            else:
                self.cands.append([p for p in range(self.N) if self.cat_of[p] == self.pattern[pos]
                                   and p not in (self.start, self.end)])
        # relaxations ------------------------------------------------------------------------
        H = 24
        dmin = [min(self.dur5[h][u] for h in range(H)) for u in range(self.N)]
        self.lbt = [dict() for _ in range(self.n)]       # lower bound on remaining time (incl. visit at u)
        self.lbt[-1] = {self.end: dmin[self.end]}
        for pos in range(self.n - 2, -1, -1):
            for u in self.cands[pos]:
                best = INF
                for v in self.cands[pos + 1]:
                    if v != u:
                        best = min(best, self.tt5[u][v] + self.lbt[pos + 1][v])
                self.lbt[pos][u] = dmin[u] + best if best < INF else INF
        if self.mode == "plaus":
            cmin = lambda u, v: min(self.nll[h][u][v] for h in range(H)) + self.lam * self.tt5[u][v]
            self.h = [dict() for _ in range(self.n)]
            self.h[-1] = {self.end: 0.0}
            for pos in range(self.n - 2, -1, -1):
                for u in self.cands[pos]:
                    best = INF
                    for v in self.cands[pos + 1]:
                        if v != u:
                            best = min(best, cmin(u, v) + self.h[pos + 1][v])
                    self.h[pos][u] = best
        else:  # time mode: h = lbt for non-final positions, 0 at the final one (terminal duration is paid in g)
            self.h = [dict(d) for d in self.lbt]
            self.h[-1] = {self.end: 0.0}

    def hour(self, e):
        return ((self.T0 + e) // 60) % 24


@dataclass
class SearchResult:
    status: str                    # ok | infeasible | capped
    path: list = field(default_factory=list)
    arrivals: list = field(default_factory=list)
    durs: list = field(default_factory=list)
    cost: float = INF
    total_time: int = None
    expansions: int = 0
    generated: int = 0
    runtime_s: float = 0.0


def evaluate_path(P: SearchProblem, path):
    """Re-simulate a full path with the problem's dynamics. Returns (cost, arrivals, durs, total_time)."""
    a, cost, arr, durs = 0, 0.0, [0], []
    for pos in range(P.n - 1):
        u, v = path[pos], path[pos + 1]
        d = P.dur5[P.hour(a)][u]
        dep = a + d
        t = P.tt5[u][v]
        cost += P.nll[P.hour(dep)][u][v] + P.lam * t if P.mode == "plaus" else 0.0
        durs.append(d)
        a = dep + t
        arr.append(a)
    dend = P.dur5[P.hour(a)][path[-1]]
    durs.append(dend)
    total = a + dend
    if P.mode == "time":
        cost = total
    return cost, arr, durs, total


def _finish(P, res, path, t0):
    cost, arr, durs, total = evaluate_path(P, path)
    res.status, res.path, res.arrivals, res.durs, res.cost, res.total_time = "ok", list(path), arr, durs, cost, total
    res.runtime_s = time.perf_counter() - t0
    return res


def best_first(P: SearchProblem, use_h: bool, max_expansions=200_000) -> SearchResult:
    """UCS (use_h=False) or A* (use_h=True)."""
    t0 = time.perf_counter()
    res = SearchResult("infeasible")
    n, B = P.n, P.budget
    if P.h[0].get(P.start, INF) == INF or (B is not None and P.lbt[0][P.start] > B):
        res.runtime_s = time.perf_counter() - t0
        return res
    nodes = [(-1, P.start, 0)]                       # (parent id, place, arrival elapsed)
    heap = [((P.h[0][P.start] if use_h else 0.0), 0, 0.0, 0, 0, 1 << P.start | 1 << P.end)]
    # heap item: (f, tiebreak, g, node id, pos, visited mask)
    closed, cnt = set(), 1
    plaus = P.mode == "plaus"
    while heap:
        f, _, g, nid, pos, vis = heapq.heappop(heap)
        _, u, a = nodes[nid]
        key = (pos, u, vis, a)
        if key in closed:
            continue
        closed.add(key)
        if pos == n - 1:
            path, k = [], nid
            while k != -1:
                path.append(nodes[k][1])
                k = nodes[k][0]
            return _finish(P, res, path[::-1], t0)
        res.expansions += 1
        if res.expansions > max_expansions:
            res.status = "capped"
            res.runtime_s = time.perf_counter() - t0
            return res
        d = P.dur5[P.hour(a)][u]
        dep = a + d
        nllrow = P.nll[P.hour(dep)][u]
        ttrow = P.tt5[u]
        last = pos + 1 == n - 1
        for v in P.cands[pos + 1]:
            if (vis >> v) & 1 and not (last and v == P.end):
                continue
            if v == u:
                continue
            t = ttrow[v]
            av = dep + t
            if B is not None:
                if last:
                    if av + P.dur5[P.hour(av)][v] > B:
                        continue
                elif av + P.lbt[pos + 1][v] > B:
                    continue
            if plaus:
                gn = g + nllrow[v] + P.lam * t
            else:
                gn = av + (P.dur5[P.hour(av)][v] if last else 0)
            hn = P.h[pos + 1][v] if use_h else 0.0
            if hn == INF or gn == INF:
                continue
            nodes.append((nid, v, av))
            cnt += 1
            res.generated += 1
            heapq.heappush(heap, (gn + hn, cnt, gn, len(nodes) - 1, pos + 1, vis | (1 << v)))
    res.runtime_s = time.perf_counter() - t0
    return res


def ucs(P, max_expansions=200_000):
    return best_first(P, False, max_expansions)


def astar(P, max_expansions=200_000):
    return best_first(P, True, max_expansions)


def greedy(P: SearchProblem) -> SearchResult:
    """Myopic greedy: at every position take the cheapest next edge that keeps the budget reachable."""
    t0 = time.perf_counter()
    res = SearchResult("infeasible")
    B = P.budget
    if B is not None and P.lbt[0][P.start] > B:
        res.runtime_s = time.perf_counter() - t0
        return res
    path, a, vis = [P.start], 0, 1 << P.start | 1 << P.end
    for pos in range(P.n - 1):
        u = path[-1]
        d = P.dur5[P.hour(a)][u]
        dep = a + d
        best, bv, bav = INF, None, None
        res.expansions += 1
        last = pos + 1 == P.n - 1
        for v in P.cands[pos + 1]:
            if ((vis >> v) & 1 and not (last and v == P.end)) or v == u:
                continue
            t = P.tt5[u][v]
            av = dep + t
            if B is not None:
                if last:
                    if av + P.dur5[P.hour(av)][v] > B:
                        continue
                elif av + P.lbt[pos + 1][v] > B:
                    continue
            c = P.nll[P.hour(dep)][u][v] + P.lam * t
            res.generated += 1
            if c < best:
                best, bv, bav = c, v, av
        if bv is None:
            res.runtime_s = time.perf_counter() - t0
            return res                                # dead end: greedy never backtracks
        path.append(bv)
        vis |= 1 << bv
        a = bav
    return _finish(P, res, path, t0)


def brute_force(P: SearchProblem) -> SearchResult:
    """Enumerate every assignment consistent with the pattern; feasibility checked on complete plans."""
    t0 = time.perf_counter()
    res = SearchResult("infeasible")
    mids = [P.cands[pos] for pos in range(1, P.n - 1)]
    best = (INF, None)
    for combo in itertools.product(*mids):
        if len(set(combo)) != len(combo):
            continue
        path = [P.start, *combo, P.end]
        res.generated += 1
        cost, _, _, total = evaluate_path(P, path)
        if P.budget is not None and total > P.budget:
            continue
        if cost < best[0]:
            best = (cost, path)
    if best[1] is None:
        res.runtime_s = time.perf_counter() - t0
        return res
    return _finish(P, res, best[1], t0)
