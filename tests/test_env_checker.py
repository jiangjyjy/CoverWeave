from coverage_repro.baselines import bfs_oracle, component_oracle
from coverage_repro.checker import check_trace
from coverage_repro.env import CoverageEnv, MergeAction
from coverage_repro.graph import CoverageGraph, Node


def test_checker_replays_ordered_merges_exactly():
    graph = CoverageGraph.complete(4)
    goal = Node(Node(0, 1), Node(2, 3))
    actions = [MergeAction(0, 1), MergeAction(1, 2), MergeAction(0, 1)]
    result = check_trace(graph, goal, actions)
    assert result.valid is True
    assert result.final_tree == goal
    assert result.steps == 3


def test_checker_accepts_cross_component_actions():
    graph = CoverageGraph(n=4, edges={(0, 1), (2, 3)})
    goal = Node(Node(0, 2), Node(1, 3))
    actions = [MergeAction(0, 1), MergeAction(1, 2), MergeAction(0, 1)]
    result = check_trace(graph, goal, actions)
    assert result.valid is True
    assert result.final_tree == goal


def test_environment_exposes_every_distinct_inventory_pair():
    graph = CoverageGraph(n=4, edges={(0, 1), (2, 3)})
    actions = CoverageEnv(graph).valid_actions(CoverageEnv(graph).initial_state())
    assert {(action.left_index, action.right_index) for action in actions} == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }


def test_bfs_solves_any_goal_on_a_disconnected_cooccurrence_graph():
    graph = CoverageGraph(n=4, edges={(0, 1), (2, 3)})
    goal = Node(Node(0, 2), Node(1, 3))
    result = bfs_oracle(graph, goal)
    assert result.success is True
    assert check_trace(graph, goal, result.actions).valid is True
    assert component_oracle(graph, goal) is True
