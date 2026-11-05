"""Category-pattern CSP.

Variables   X_0 .. X_{n-1}: the category of each stop.  X_0 / X_{n-1} are fixed to the category of the
            start / end place (unary constraints), so the domain of those two variables is a singleton.
Constraints (1) neighbouring stops have different categories                        (binary, !=)
            (2) the must-see category appears at least once                         (global)
            (3) no category appears more than twice, and never more often than the
                number of distinct places of that category in the pool              (global, "cap")
            (4) expected total time  sum_i dur(X_i) + (n-1)*hop  <=  budget          (global, sum)

Three solvers share one backtracking skeleton and differ only in the *inference* done after each
assignment:
    'bt'   plain backtracking: only checks constraints among already-assigned variables
    'fc'   + forward checking: prunes the domains of unassigned variables
    'ac3'  + AC-3 on the binary '!=' constraints, run after FC (i.e. maintaining arc consistency)

All three are sound (they never remove a value that is part of a solution), so they must return exactly
the same set of solutions; the unit tests check this against brute force.
"""
import time
from collections import deque
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class CSPSpec:
    n: int
    cats: list                 # category names (domain of the middle variables)
    first: str                 # category of the start place
    last: str                  # category of the end place
    must: str                  # must-see category
    cap: dict                  # max occurrences per category (already min(2, pool size))
    dur: dict                  # expected visit minutes per category
    dur_first: float           # expected minutes at the start place
    dur_last: float            # expected minutes at the end place
    hop: float                 # expected travel minutes between consecutive stops
    budget: float = None       # minutes (None = no time constraint)

    def dur_of(self, i, c):
        if i == 0:
            return self.dur_first
        if i == self.n - 1:
            return self.dur_last
        return self.dur[c]

    def total_time(self, pattern):
        return sum(self.dur_of(i, c) for i, c in enumerate(pattern)) + (self.n - 1) * self.hop


def is_solution(spec: CSPSpec, p) -> bool:
    """Independent checker used by the tests (does not share code with the solvers)."""
    if len(p) != spec.n or p[0] != spec.first or p[-1] != spec.last:
        return False
    if any(p[i] == p[i + 1] for i in range(spec.n - 1)):
        return False
    if spec.must not in p:
        return False
    if any(p.count(c) > spec.cap[c] for c in set(p)):
        return False
    if spec.budget is not None and spec.total_time(p) > spec.budget + 1e-9:
        return False
    return True


@dataclass
class CSPStats:
    nodes: int = 0          # (variable, value) assignments attempted
    backtracks: int = 0     # failed nodes (dead ends)
    checks: int = 0         # elementary constraint / domain checks
    solutions: int = 0
    time_ms: float = 0.0
    truncated: bool = False


# --------------------------------------------------------------------------- AC-3
def revise_neq(domains, a, b, stats):
    """Remove from D(a) every value with no support in D(b) under  a != b."""
    removed = False
    stats.checks += len(domains[a])
    if len(domains[b]) == 1:
        (y,) = tuple(domains[b])
        if y in domains[a]:
            domains[a].discard(y)
            removed = True
    return removed


def ac3(domains, stats, arcs=None):
    """Enforce arc consistency on the chain of '!=' constraints. Returns False on domain wipe-out."""
    n = len(domains)
    if arcs is None:
        arcs = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    q = deque(arcs)
    while q:
        a, b = q.popleft()
        if revise_neq(domains, a, b, stats):
            if not domains[a]:
                return False
            for k in (a - 1, a + 1):
                if 0 <= k < n and k != b:
                    q.append((k, a))
    return True


# --------------------------------------------------------------------------- solver
def solve(spec: CSPSpec, method="fc", max_solutions=None, order_fn=None, node_limit=None):
    """Enumerate valid category patterns.

    order_fn(i, prev_category, domain) -> list of categories, lets the caller try plausible values first.
    Returns (solutions, stats).
    """
    assert method in ("bt", "fc", "ac3")
    t0 = time.perf_counter()
    n, st = spec.n, CSPStats()
    budget = INF if spec.budget is None else spec.budget
    sols = []

    def finish():
        st.time_ms = (time.perf_counter() - t0) * 1000
        return sols, st

    if spec.must not in spec.cats or n < 2:
        return finish()

    def dmin(i, dom):
        return min(spec.dur_of(i, c) for c in dom)

    def order(i, prev, dom):
        return order_fn(i, prev, dom) if order_fn else sorted(dom)

    # ---- inference after assigning X_i = v (forward checking, optionally + AC-3) ---------------
    def infer(i, v, doms, counts, tsum):
        # the fixed end category is "reserved": it will be used once by X_{n-1}
        res = 1 if i < n - 1 else 0                           # reservation ends once X_{n-1} is assigned
        eff = lambda c: counts.get(c, 0) + (res if c == spec.last else 0)
        st.checks += 1
        if eff(v) > spec.cap[v]:
            return False
        for j in range(i + 1, n):
            d = doms[j]
            if j == i + 1:
                st.checks += 1
                d.discard(v)                                   # neighbouring stops differ
            if j < n - 1 and eff(v) >= spec.cap[v]:
                st.checks += 1
                d.discard(v)                                   # category used up
            if not d:
                return False
        if budget < INF and i + 1 < n:                         # time bound with cheapest completion
            mins = {j: dmin(j, doms[j]) for j in range(i + 1, n)}
            tot_min = sum(mins.values())
            for j in range(i + 1, n):
                for c in list(doms[j]):
                    st.checks += 1
                    if tsum + spec.dur_of(j, c) + (tot_min - mins[j]) + (n - 1) * spec.hop > budget + 1e-9:
                        doms[j].discard(c)
                if not doms[j]:
                    return False
        if counts.get(spec.must, 0) == 0 and not any(spec.must in doms[j] for j in range(i + 1, n)):
            return False                                       # must-see can no longer appear
        if method == "ac3" and i + 1 < n:
            doms[i] = {v}
            if not ac3(doms, st, arcs=[(i, i + 1), (i + 1, i)]):
                return False
        return True

    # ---- recursive backtracking --------------------------------------------------------------
    def rec(i, assign, counts, tsum, doms):
        if i == n:
            if counts.get(spec.must, 0) > 0:                   # constraint 2 (leaf check for BT)
                sols.append(tuple(assign))
                st.solutions += 1
            return max_solutions is not None and len(sols) >= max_solutions
        prev = assign[-1] if assign else None
        for v in order(i, prev, doms[i]):
            if node_limit is not None and st.nodes >= node_limit:
                st.truncated = True
                return True
            st.nodes += 1
            st.checks += 3
            t_new = tsum + spec.dur_of(i, v)
            if (prev is not None and v == prev) or counts.get(v, 0) + 1 > spec.cap[v] \
                    or t_new + i * spec.hop > budget + 1e-9:
                st.backtracks += 1
                continue
            counts[v] = counts.get(v, 0) + 1
            assign.append(v)
            nd, ok = doms, True
            if method != "bt":
                nd = [set(d) for d in doms]
                ok = infer(i, v, nd, counts, t_new)
            stop = False
            if ok:
                stop = rec(i + 1, assign, counts, t_new, nd)
            else:
                st.backtracks += 1
            assign.pop()
            counts[v] -= 1
            if stop:
                return True
        return False

    doms0 = [set(spec.cats) for _ in range(n)]
    doms0[0], doms0[n - 1] = {spec.first}, {spec.last}
    if method != "bt":
        # forward-check from the fixed last variable (X_0 is handled by the normal recursion)
        doms0[n - 2].discard(spec.last)
        if method == "ac3" and not ac3(doms0, st):
            return finish()
        if any(not d for d in doms0):
            return finish()
    rec(0, [], {}, 0.0, doms0)
    return finish()
