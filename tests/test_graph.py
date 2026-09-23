from coverage_repro.graph import CoverageGraph, make_goal, tree_depth


def test_random_graph_uses_exact_edge_count_and_density():
    graph = CoverageGraph.random(n=12, lam=1.5, seed=7)
    assert graph.m == round(1.5 * 12 / 2)
    assert graph.rho == 2 * graph.m / (12 * 11)
    assert len(graph.edges) == graph.m


def test_goal_is_ordered_binary_tree_with_requested_feasible_depth():
    goal = make_goal(n_leaves=7, requested_depth=3, seed=2)
    assert tree_depth(goal) == 3
    assert sorted(goal.leaves()) == list(range(7))
