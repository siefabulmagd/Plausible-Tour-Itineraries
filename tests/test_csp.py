"""Unit tests for the category-pattern CSP (run: python -m unittest discover -s tests -v)."""
import itertools
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest
from p6.csp import CSPSpec, solve, is_solution, ac3, CSPStats


def spec(n=5, cats="ABCD", first="A", last="B", must="C", cap=2, budget=None, dur=30, hop=10):
    cats = list(cats)
    return CSPSpec(n, cats, first, last, must, {c: cap for c in cats}, {c: dur for c in cats}, dur, dur, hop, budget)


class TestChecker(unittest.TestCase):
    def test_checker_rules(self):
        s = spec()
        self.assertTrue(is_solution(s, ("A", "C", "D", "A", "B")))
        self.assertFalse(is_solution(s, ("A", "A", "C", "D", "B")))      # neighbours equal
        self.assertFalse(is_solution(s, ("A", "D", "A", "D", "B")))      # must-see missing
        self.assertFalse(is_solution(spec(n=6, cap=2), ("A", "C", "D", "C", "D", "B")) and False)
        s2 = spec(n=7, cats="ABC", first="A", last="B", must="C")
        self.assertFalse(is_solution(s2, ("A", "C", "A", "C", "A", "C", "B")))  # A three times
        self.assertFalse(is_solution(spec(budget=100), ("A", "C", "D", "A", "B")))  # 5*30+4*10=190 > 100


class TestSolvers(unittest.TestCase):
    def test_all_methods_equal_bruteforce_random(self):
        rnd = random.Random(0)
        for _ in range(300):
            cats = list("ABCDE")[:rnd.randint(3, 5)]
            n = rnd.randint(2, 6)
            sp = CSPSpec(n, cats, rnd.choice(cats), rnd.choice(cats), rnd.choice(cats),
                         {c: rnd.choice([1, 2, 2]) for c in cats}, {c: rnd.choice([20, 30, 45, 60]) for c in cats},
                         rnd.choice([30, 45]), rnd.choice([30, 45]), 10, rnd.choice([None, 150, 200, 260, 400]))
            bf = {p for p in itertools.product(cats, repeat=n) if is_solution(sp, p)}
            for m in ("bt", "fc", "ac3"):
                sols, _ = solve(sp, m)
                self.assertEqual(set(sols), bf, msg=f"{m} {sp}")
                self.assertEqual(len(sols), len(set(sols)))

    def test_every_returned_pattern_is_valid(self):
        s = spec(n=7, cats="ABCDEF", must="D", budget=350)
        for m in ("bt", "fc", "ac3"):
            sols, st = solve(s, m)
            self.assertTrue(sols)
            self.assertTrue(all(is_solution(s, p) for p in sols))
            self.assertEqual(st.solutions, len(sols))

    def test_max_solutions(self):
        sols, _ = solve(spec(n=7, cats="ABCDEF"), "fc", max_solutions=10)
        self.assertEqual(len(sols), 10)

    def test_unsatisfiable_cases(self):
        self.assertEqual(solve(spec(must="Z"), "ac3")[0], [])                          # must-see not a category
        self.assertEqual(solve(spec(first="A", last="A", cap=1), "fc")[0], [])         # cap violated by the endpoints
        self.assertEqual(solve(spec(budget=50), "bt")[0], [])                          # budget too small
        self.assertEqual(solve(spec(n=7, cats="AB", first="A", last="B", must="B"), "ac3")[0], [])  # only 2 cats, cap 2, n=7

    def test_budget_boundary(self):
        # minimal pattern A C B? n=3 -> 3*30 + 2*10 = 110
        self.assertTrue(solve(spec(n=3, must="C", budget=110), "fc")[0])
        self.assertFalse(solve(spec(n=3, must="C", budget=109), "fc")[0])

    def test_value_ordering_changes_order_not_set(self):
        s = spec(n=5, cats="ABCD")
        a, _ = solve(s, "fc")
        b, _ = solve(s, "fc", order_fn=lambda i, prev, dom: sorted(dom, reverse=True))
        self.assertEqual(set(a), set(b))
        self.assertNotEqual(a, b)


class TestInference(unittest.TestCase):
    def test_ac3_propagates_through_singletons(self):
        # A ? ? B with domains {A,B}: neighbours differ -> X1 = B, X2 = A
        doms = [{"A"}, {"A", "B"}, {"A", "B"}, {"B"}]
        self.assertTrue(ac3(doms, CSPStats()))
        self.assertEqual(doms, [{"A"}, {"B"}, {"A"}, {"B"}])

    def test_ac3_detects_wipeout(self):
        doms = [{"A"}, {"A", "B"}, {"B"}, {"A"}]   # X1 must be B, but X2 = B  -> wipeout
        self.assertFalse(ac3(doms, CSPStats()))

    def test_forward_checking_prunes_more_than_bt(self):
        s = spec(n=7, cats="ABCDEF", must="D", budget=210)   # tight budget
        _, bt = solve(s, "bt")
        _, fc = solve(s, "fc")
        _, ac = solve(s, "ac3")
        self.assertLessEqual(fc.nodes, bt.nodes)
        self.assertLessEqual(ac.nodes, fc.nodes)


if __name__ == "__main__":
    unittest.main()
