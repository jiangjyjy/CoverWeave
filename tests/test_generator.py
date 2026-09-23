from coverage_repro.checker import check_trace
from coverage_repro.generator import generate_trace
from coverage_repro.graph import CoverageGraph, Node


def test_oracle_generator_trace_is_checker_valid():
    graph = CoverageGraph.complete(4)
    goal = Node(Node(0, 1), Node(2, 3))
    trace = generate_trace(graph, goal, policy="oracle", seed=3)
    assert trace.success is True
    assert check_trace(graph, goal, trace.actions).valid is True


def test_random_generator_has_a_finite_action_budget():
    graph = CoverageGraph.complete(5)
    goal = Node(Node(Node(0, 1), 2), Node(3, 4))
    trace = generate_trace(graph, goal, policy="random", seed=3, budget=6)
    assert trace.budget == 6
    assert len(trace.actions) <= trace.budget
    assert trace.attempts <= trace.budget
