"""Unit tests for the itinerary search (UCS / A* / Greedy) against brute force on small pools."""
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest
from p6.search import SearchProblem, ucs, astar, greedy, brute_force, evaluate_path, INF


def make(seed, N=9, C=3, n=5, budget="none", mode="plaus"):
    r = random.Random(seed)
    cat_of = [i % C for i in range(N)]
    nll = [[[INF if u == v else r.uniform(0.1, 6) for v in range(N)] for u in range(N)] for _ in range(24)]
    dur5 = [[5 * r.randint(2, 12) for _ in range(N)] for _ in range(24)]
    tt5 = [[5 * r.randint(1, 6) for _ in range(N)] for _ in range(N)]
    start, end = 0, N - 2 if (N - 1) % C == 0 else N - 1
    pat = [cat_of[start]]
    for _ in range(1, n - 1):
        pat.append(r.choice([c for c in range(C) if c != pat[-1]]))
    pat.append(cat_of[end])
    T0, lam = r.choice([480, 540, 780, 1000]), r.choice([0.0, 0.05, 0.2])
    ref = brute_force(SearchProblem(pat, start, end, cat_of, nll, dur5, tt5, T0, lam, None, mode))
    if ref.status != "ok":
        return None
    B = None if budget == "none" else int(ref.total_time * 0.85) if budget == "tight" else int(ref.total_time * 1.2)
    return SearchProblem(pat, start, end, cat_of, nll, dur5, tt5, T0, lam, B, mode)


class TestSearch(unittest.TestCase):
    def check(self, budget, mode):
        n_checked = 0
        for seed in range(60):
            P = make(seed, n=random.Random(seed).choice([3, 4, 5, 6]), budget=budget, mode=mode)
            if P is None:
                continue
            bf, u, a, g = brute_force(P), ucs(P), astar(P), greedy(P)
            if bf.status != "ok":
                self.assertEqual((u.status, a.status), ("infeasible", "infeasible"))
                continue
            n_checked += 1
            self.assertAlmostEqual(u.cost, bf.cost, places=9, msg=f"UCS seed {seed}")
            self.assertAlmostEqual(a.cost, bf.cost, places=9, msg=f"A* seed {seed}")
            if P.budget is not None:
                self.assertLessEqual(u.total_time, P.budget)
                self.assertLessEqual(a.total_time, P.budget)
            if g.status == "ok":
                self.assertGreaterEqual(g.cost, bf.cost - 1e-9)           # greedy can never beat the optimum
            self.assertLessEqual(a.expansions, u.expansions + 1)           # a good heuristic never hurts
        self.assertGreater(n_checked, 10)

    def test_unconstrained_plaus(self): self.check("none", "plaus")
    def test_loose_budget_plaus(self): self.check("loose", "plaus")
    def test_tight_budget_plaus(self): self.check("tight", "plaus")
    def test_time_mode(self): self.check("none", "time")

    def test_infeasible_budget(self):
        P = make(1, budget="none")
        P.budget = 5
        for f in (ucs, astar, greedy, brute_force):
            self.assertEqual(f(P).status, "infeasible")

    def test_heuristic_is_admissible_and_consistent(self):
        for seed in range(30):
            P = make(seed, n=5)
            if P is None:
                continue
            res = brute_force(P)
            self.assertLessEqual(P.h[0][P.start], res.cost + 1e-9)           # admissible at the root
            for pos in range(P.n - 1):                                         # consistent on every edge
                for u in P.cands[pos]:
                    for v in P.cands[pos + 1]:
                        if u == v:
                            continue
                        cmin = min(P.nll[h][u][v] for h in range(24)) + P.lam * P.tt5[u][v]
                        self.assertLessEqual(P.h[pos][u], cmin + P.h[pos + 1][v] + 1e-9)

    def test_plan_respects_pattern_and_no_repeats(self):
        P = make(5, n=6)
        r = astar(P)
        self.assertEqual([P.cat_of[p] for p in r.path], P.pattern)
        self.assertEqual(len(set(r.path)), len(r.path))
        self.assertEqual((r.path[0], r.path[-1]), (P.start, P.end))
        # arrival times are consistent with the dynamics
        _, arr, durs, total = evaluate_path(P, r.path)
        self.assertEqual((arr, durs, total), (r.arrivals, r.durs, r.total_time))


if __name__ == "__main__":
    unittest.main()
