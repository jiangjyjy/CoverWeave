"""Goal-blind CompForge v3 baselines, SFT, and informativeness gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from .compforge import Rule, RuleAction, RuleKind, check_compforge_trace, uniform_cost_oracle
from .compforge_encoding import CompForgeActionCodec, CompForgeTokenCodec, collate_compforge_batch, encode_compforge_source
from .compforge_v3 import environment_from_v3_row, goal_blind_legal_action_mask, inventory_from_v3_row
from .model_sft import MergeTransformer, TransformerConfig
from .train_sft import seed_everything


@dataclass(frozen=True)
class V3Codecs:
    token: CompForgeTokenCodec
    action: CompForgeActionCodec
    model_config: TransformerConfig


def build_v3_codecs(
    rows: Sequence[Mapping[str, object]],
    *,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 64,
) -> V3Codecs:
    """Fit source vocabulary from public observations only."""
    if not rows:
        raise ValueError("rows must not be empty")
    if d_model < 1 or nhead < 1 or d_model % nhead:
        raise ValueError("d_model must be positive and divisible by nhead")
    reference_row = max(rows, key=lambda row: len(row["instance"]["rules"]))
    env = environment_from_v3_row(reference_row)
    token = CompForgeTokenCodec.from_rows(rows)
    action = CompForgeActionCodec.from_env(env, max(len(row["inventory"]) for row in rows))
    model_config = TransformerConfig(
        src_vocab_size=token.vocab_size,
        action_vocab_size=action.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_src_len=max(len(encode_compforge_source(row, token)) for row in rows),
        max_target_len=max(len(row["actions"]) + 2 for row in rows),
    )
    return V3Codecs(token, action, model_config)


def _episode_metrics(
    row: Mapping[str, object],
    actions: Sequence[RuleAction],
    legal_counts: Sequence[int],
    failure_reason: str,
) -> dict[str, object]:
    env = environment_from_v3_row(row)
    inventory = inventory_from_v3_row(row)
    checked = check_compforge_trace(env, inventory, str(row["goal"]), tuple(actions))
    oracle = uniform_cost_oracle(env, inventory, str(row["goal"]), max_expansions=1_000)
    reachable = uniform_cost_oracle(env, checked.final_state.inventory, str(row["goal"]), max_expansions=1_000).success
    return {
        "success": checked.valid,
        "valid_action_rate": 1.0,
        "mean_legal_branching": sum(legal_counts) / len(legal_counts) if legal_counts else 0.0,
        "dead_end": not checked.valid and not reachable,
        "cost_ratio": checked.total_cost / oracle.total_cost if checked.valid and oracle.success else None,
        "failure_reason": "none" if checked.valid else failure_reason,
        "steps": checked.steps,
    }


def _aggregate(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(episodes)
    failures = Counter(str(item["failure_reason"]) for item in episodes if not item["success"])
    costs = [float(item["cost_ratio"]) for item in episodes if item["cost_ratio"] is not None]
    return {
        "exact_solve_rate": sum(bool(item["success"]) for item in episodes) / total if total else 0.0,
        "valid_action_rate": sum(float(item["valid_action_rate"]) for item in episodes) / total if total else 0.0,
        "mean_legal_branching": sum(float(item["mean_legal_branching"]) for item in episodes) / total if total else 0.0,
        "dead_end_rate": sum(bool(item["dead_end"]) for item in episodes) / total if total else 0.0,
        "cost_ratio": sum(costs) / len(costs) if costs else None,
        "failure_counts": dict(sorted(failures.items())),
        "episodes": list(episodes),
    }


def evaluate_oracle(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    episodes = []
    for row in rows:
        actions = tuple(
            RuleAction(str(value["rule_id"]), tuple(int(slot) for slot in value["operand_slots"]))
            for value in row["oracle_actions"]
        )
        episodes.append(_episode_metrics(row, actions, row["legal_action_counts"], "oracle_replay_failed"))
    return _aggregate(episodes)


def evaluate_random_legal(
    rows: Sequence[Mapping[str, object]],
    *,
    codecs: V3Codecs,
    seed: int,
    action_budget: int,
) -> dict[str, object]:
    """Sample uniformly from EOS plus the current goal-blind legal action tokens."""
    rng = Random(seed)
    episodes = []
    for row in rows:
        env = environment_from_v3_row(row)
        inventory = inventory_from_v3_row(row)
        state = env.initial_state(inventory)
        actions: list[RuleAction] = []
        counts: list[int] = []
        reason = "budget_exhausted"
        for _ in range(action_budget):
            legal = tuple(env.valid_actions(state))
            counts.append(len(legal))
            mask = goal_blind_legal_action_mask(env, state, codecs.action)
            enabled_tokens = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            if not enabled_tokens:
                reason = "no_legal_action"
                break
            token = rng.choice(enabled_tokens)
            if token == codecs.action.eos_token:
                reason = "early_eos"
                break
            action = codecs.action.decode(token)
            if action is None or action not in legal:
                reason = "malformed_action"
                break
            actions.append(action)
            state = env.apply(state, action)
        episodes.append(_episode_metrics(row, actions, counts, reason))
    return _aggregate(episodes)


def decode_v3_episode(
    model: torch.nn.Module,
    row: Mapping[str, object],
    codecs: V3Codecs,
    device: str | torch.device,
    action_budget: int,
) -> dict[str, object]:
    """Greedy Transformer decoding with a mask that has no goal argument."""
    target = torch.device(device)
    env = environment_from_v3_row(row)
    inventory = inventory_from_v3_row(row)
    state = env.initial_state(inventory)
    source = torch.tensor(encode_compforge_source(row, codecs.token), dtype=torch.long, device=target).unsqueeze(0)
    tokens = [codecs.action.bos_token]
    actions: list[RuleAction] = []
    counts: list[int] = []
    reason = "budget_exhausted"
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(action_budget):
                decoder = torch.tensor(tokens, dtype=torch.long, device=target).unsqueeze(0)
                logits = model(source, decoder, source.eq(codecs.token.PAD), decoder.eq(codecs.action.pad_token))
                legal = tuple(env.valid_actions(state))
                counts.append(len(legal))
                mask = goal_blind_legal_action_mask(env, state, codecs.action).to(target)
                if not mask.any():
                    reason = "no_legal_action"
                    break
                token = int(logits[0, -1].masked_fill(~mask, float("-inf")).argmax().item())
                tokens.append(token)
                if token == codecs.action.eos_token:
                    reason = "early_eos"
                    break
                action = codecs.action.decode(token)
                if action is None or action not in legal:
                    reason = "malformed_action"
                    break
                actions.append(action)
                state = env.apply(state, action)
    finally:
        model.train(was_training)
    return _episode_metrics(row, actions, counts, reason)


def evaluate_untrained_transformer(
    rows: Sequence[Mapping[str, object]],
    *,
    codecs: V3Codecs,
    seed: int,
    device: str | torch.device,
    action_budget: int,
) -> dict[str, object]:
    seed_everything(seed)
    model = MergeTransformer(codecs.model_config).to(torch.device(device))
    return _aggregate([decode_v3_episode(model, row, codecs, device, action_budget) for row in rows])


def _checkpoint_payload(model: MergeTransformer, codecs: V3Codecs, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    return {
        "model_config": asdict(codecs.model_config),
        "token_states": list(codecs.token.states),
        "rules": [
            {"rule_id": rule.rule_id, "kind": rule.kind.value, "inputs": list(rule.inputs), "outputs": list(rule.outputs), "cost": rule.cost}
            for rule in codecs.action.rules
        ],
        "max_inventory": codecs.action.max_inventory,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def load_v3_sft_checkpoint(path: str | Path, device: str | torch.device) -> tuple[MergeTransformer, V3Codecs]:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    rules = tuple(
        Rule(str(item["rule_id"]), RuleKind(str(item["kind"])), tuple(item["inputs"]), tuple(item["outputs"]), int(item["cost"]))
        for item in payload["rules"]
    )
    codecs = V3Codecs(
        CompForgeTokenCodec(tuple(payload["token_states"])),
        CompForgeActionCodec(rules, int(payload["max_inventory"])),
        TransformerConfig(**dict(payload["model_config"])),
    )
    model = MergeTransformer(codecs.model_config).to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, codecs


def evaluate_checkpoint_sft(
    checkpoint: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    device: str | torch.device,
    action_budget: int,
) -> dict[str, object]:
    model, codecs = load_v3_sft_checkpoint(checkpoint, device)
    return _aggregate(
        [decode_v3_episode(model, row, codecs, device, action_budget) for row in rows]
    )


def evaluate_trained_sft(
    train_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
    *,
    codecs: V3Codecs,
    seed: int,
    device: str | torch.device,
    run_dir: str | Path,
    epochs: int,
    batch_size: int,
    action_budget: int,
) -> dict[str, object]:
    """Train a Transformer on train actions, reload it, then evaluate the reload."""
    if not train_rows or epochs < 1 or batch_size < 1:
        raise ValueError("train_rows, epochs, and batch_size must be positive")
    seed_everything(seed)
    target = torch.device(device)
    model = MergeTransformer(codecs.model_config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(train_rows), batch_size):
            batch = {
                name: value.to(target)
                for name, value in collate_compforge_batch(train_rows[start:start + batch_size], codecs.token, codecs.action).items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["src"], batch["decoder_input"], batch["src_padding_mask"], batch["decoder_padding_mask"])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["targets"].reshape(-1),
                ignore_index=codecs.action.pad_token,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite v3 SFT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "v3-sft.pt"
    torch.save(_checkpoint_payload(model, codecs, optimizer), checkpoint)
    metrics = evaluate_checkpoint_sft(
        checkpoint,
        eval_rows,
        device=target,
        action_budget=action_budget,
    )
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def run_v3_gate(
    per_split: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    required_splits = ("valid", "held_out_same_depth", "held_out_deep")
    missing = set(required_splits) - set(per_split)
    if missing:
        raise ValueError(f"gate is missing required splits: {sorted(missing)}")
    conditions = {
        "oracle_all_splits": all(
            float(per_split[split]["oracle"]["exact_solve_rate"]) == 1.0
            for split in required_splits
        ),
        "valid_trained_nonzero": (
            float(per_split["valid"]["trained"]["exact_solve_rate"]) > 0.0
        ),
    }
    for split in ("held_out_same_depth", "held_out_deep"):
        trained = float(per_split[split]["trained"]["exact_solve_rate"])
        random = float(per_split[split]["random"]["exact_solve_rate"])
        untrained = float(per_split[split]["untrained"]["exact_solve_rate"])
        conditions[f"{split}_trained_beats_random"] = trained >= random + 0.20
        conditions[f"{split}_trained_beats_untrained"] = trained >= untrained + 0.20
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "per_split": {split: dict(metrics) for split, metrics in per_split.items()},
    }
