from coverage_repro.baselines import bfs_oracle, component_oracle
from coverage_repro.graph import CoverageGraph, Node


def test_bfs_oracle_is_perfect_on_a_connected_graph():
    graph = CoverageGraph(n=4, edges={(0, 1), (1, 2), (2, 3)})
    goal = Node(Node(0, 1), Node(2, 3))
    result = bfs_oracle(graph, goal)
    assert result.success is True
    assert result.checked is True


def test_component_oracle_matches_unrestricted_merge_environment():
    graph = CoverageGraph(n=4, edges={(0, 1), (2, 3)})
    connected_goal = Node(Node(0, 1), Node(2, 3))
    disconnected_goal = Node(Node(Node(0, 1), 2), 3)
    assert component_oracle(graph, connected_goal) is True
    assert component_oracle(graph, disconnected_goal) is True
    assert component_oracle(graph, Node(0, 1)) is True
