import pytest

from coverage_repro.compforge import (
    CompForgeEnv,
    ObjectInstance,
    Rule,
    RuleAction,
    RuleKind,
    check_compforge_trace,
    uniform_cost_oracle,
)


RULES = (
    Rule("transform-0", RuleKind.TRANSFORM, ("a0",), ("t0",), 1),
    Rule("bind-01", RuleKind.BIND, ("t0", "a1"), ("b01",), 2),
    Rule("sync-23", RuleKind.SYNCHRONIZE, ("a2", "a3"), ("s2", "s3"), 3),
    Rule("synth", RuleKind.SYNTHESIZE, ("b01", "s2", "s3"), ("product",), 4),
    Rule("decompose", RuleKind.DECOMPOSE, ("product",), ("x", "y"), 5),
)


def objects(*states: str) -> tuple[ObjectInstance, ...]:
    return tuple(ObjectInstance(index, state, (index,)) for index, state in enumerate(states))


@pytest.mark.parametrize(
    ("inventory", "action", "expected_states", "expected_cost", "expected_next_id"),
    (
        (("a0",), RuleAction("transform-0", (0,)), ("t0",), 1, 2),
        (("t0", "a1"), RuleAction("bind-01", (0, 1)), ("b01",), 2, 3),
        (("a2", "a3"), RuleAction("sync-23", (0, 1)), ("s2", "s3"), 3, 4),
        (("b01", "s2", "s3"), RuleAction("synth", (0, 1, 2)), ("product",), 4, 4),
        (("product",), RuleAction("decompose", (0,)), ("x", "y"), 5, 3),
    ),
)
def test_every_rule_kind_has_a_legal_transition(
    inventory: tuple[str, ...],
    action: RuleAction,
    expected_states: tuple[str, ...],
    expected_cost: int,
    expected_next_id: int,
):
    env = CompForgeEnv(RULES)
    initial = env.initial_state(objects(*inventory))

    assert action in env.valid_actions(initial)
    result = env.apply(initial, action)

    assert tuple(item.state for item in result.inventory) == expected_states
    assert result.total_cost == expected_cost
    assert result.next_instance_id == expected_next_id


def test_apply_inserts_outputs_at_the_lowest_consumed_slot():
    env = CompForgeEnv(RULES)
    initial = env.initial_state(objects("a2", "a3", "a0", "a1"))

    result = env.apply(initial, RuleAction("sync-23", (0, 1)))

    assert tuple(item.state for item in result.inventory) == ("s2", "s3", "a0", "a1")


@pytest.mark.parametrize(
    "action",
    (
        RuleAction("bind-01", (0, 0)),
        RuleAction("transform-0", (3,)),
        RuleAction("bind-01", (1, 0)),
    ),
)
def test_apply_rejects_illegal_rule_actions(action: RuleAction):
    env = CompForgeEnv(RULES)
    initial = env.initial_state(objects("t0", "a1"))

    with pytest.raises(ValueError, match="legal"):
        env.apply(initial, action)


def test_checker_replays_a_multi_rule_trace_to_the_goal():
    env = CompForgeEnv(RULES)
    inventory = objects("a0", "a1", "a2", "a3")
    actions = (
        RuleAction("transform-0", (0,)),
        RuleAction("bind-01", (0, 1)),
        RuleAction("sync-23", (1, 2)),
        RuleAction("synth", (0, 1, 2)),
    )

    result = check_compforge_trace(env, inventory, "product", actions)

    assert result.valid is True
    assert result.steps == 4
    assert result.total_cost == 10


def test_checker_rejects_a_trace_that_does_not_produce_the_goal():
    env = CompForgeEnv(RULES)
    result = check_compforge_trace(
        env,
        objects("a0"),
        "product",
        (RuleAction("transform-0", (0,)),),
    )

    assert result.valid is False
    assert result.reason == "goal state is absent"


def test_uniform_cost_oracle_prefers_a_cheaper_two_step_trace():
    env = CompForgeEnv(
        (
            Rule("direct", RuleKind.TRANSFORM, ("a0",), ("goal",), 9),
            Rule("to-middle", RuleKind.TRANSFORM, ("a0",), ("middle",), 2),
            Rule("to-goal", RuleKind.TRANSFORM, ("middle",), ("goal",), 3),
        )
    )

    result = uniform_cost_oracle(env, objects("a0"), "goal", max_expansions=20)

    assert result.success is True
    assert result.total_cost == 5
    assert tuple(action.rule_id for action in result.actions) == ("to-middle", "to-goal")
    assert check_compforge_trace(env, objects("a0"), "goal", result.actions).valid is True


def test_uniform_cost_oracle_reports_an_expansion_limit():
    env = CompForgeEnv((Rule("direct", RuleKind.TRANSFORM, ("a0",), ("goal",), 1),))

    result = uniform_cost_oracle(env, objects("a0"), "goal", max_expansions=0)

    assert result.success is False
    assert result.reason == "expansion limit reached"
