from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from .checker import check_trace
from .encoding import ActionCodec, TokenCodec, encode_source, legal_action_mask
from .env import CoverageEnv, EnvState, MergeAction
from .graph import CoverageGraph, Tree, tree_from_dict
from .io import write_json


@dataclass(frozen=True)
class TransformerConfig:
    src_vocab_size: int
    action_vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_src_len: int = 512
    max_target_len: int = 128
    src_padding_idx: int = 0
    action_padding_idx: int = 0


class MergeTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(
            config.src_vocab_size, config.d_model, padding_idx=config.src_padding_idx
        )
        self.action_embedding = nn.Embedding(
            config.action_vocab_size, config.d_model, padding_idx=config.action_padding_idx
        )
        self.src_position = nn.Embedding(config.max_src_len, config.d_model)
        self.action_position = nn.Embedding(config.max_target_len, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.action_head = nn.Linear(config.d_model, config.action_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding, padding_idx in (
            (self.src_embedding, self.config.src_padding_idx),
            (self.action_embedding, self.config.action_padding_idx),
            (self.src_position, None),
            (self.action_position, None),
        ):
            nn.init.xavier_uniform_(embedding.weight)
            if padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[padding_idx].zero_()
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        src: Tensor,
        decoder_input: Tensor,
        src_padding_mask: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, src_length = src.shape
        _, target_length = decoder_input.shape
        if src_length > self.config.max_src_len:
            raise ValueError("source sequence exceeds configured max_src_len")
        if target_length > self.config.max_target_len:
            raise ValueError("decoder sequence exceeds configured max_target_len")

        src_positions = torch.arange(src_length, device=src.device).unsqueeze(0).expand(batch_size, -1)
        target_positions = torch.arange(target_length, device=decoder_input.device).unsqueeze(0).expand(
            batch_size, -1
        )
        src_features = self.src_embedding(src) + self.src_position(src_positions)
        target_features = self.action_embedding(decoder_input) + self.action_position(target_positions)
        decoded = self.transformer(
            src_features,
            target_features,
            tgt_mask=self._causal_mask(target_length, decoder_input.device),
            src_key_padding_mask=src_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=decoder_padding_mask,
        )
        return self.action_head(decoded)


@dataclass(frozen=True)
class CodecBundle:
    token: TokenCodec
    action: ActionCodec


Codecs = CodecBundle


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    actions: tuple[MergeAction, ...]
    legal_actions_at_step: tuple[tuple[MergeAction, ...], ...]
    valid_action_rate: float
    checker_valid: bool
    failure_reason: str
    emitted_tokens: tuple[int, ...]


def _unpack_codecs(codecs: CodecBundle | tuple[TokenCodec, ActionCodec] | Mapping[str, object]) -> CodecBundle:
    if isinstance(codecs, CodecBundle):
        return codecs
    if isinstance(codecs, (tuple, list)) and len(codecs) == 2:
        return CodecBundle(codecs[0], codecs[1])
    if isinstance(codecs, Mapping):
        token = codecs.get("token") or codecs.get("token_codec")
        action = codecs.get("action") or codecs.get("action_codec")
        if isinstance(token, TokenCodec) and isinstance(action, ActionCodec):
            return CodecBundle(token, action)
    raise TypeError("codecs must contain a TokenCodec and ActionCodec")


def _result(
    success: bool,
    actions: list[MergeAction],
    legal_actions: list[tuple[MergeAction, ...]],
    checker_valid: bool,
    failure_reason: str,
    emitted_tokens: list[int],
) -> DecodeResult:
    valid_count = sum(
        action in legal_actions[index]
        for index, action in enumerate(actions)
        if index < len(legal_actions)
    )
    rate = valid_count / len(actions) if actions else 1.0
    return DecodeResult(
        success=success,
        actions=tuple(actions),
        legal_actions_at_step=tuple(legal_actions),
        valid_action_rate=rate,
        checker_valid=checker_valid,
        failure_reason=failure_reason,
        emitted_tokens=tuple(emitted_tokens),
    )


def decode_episode(
    model: nn.Module,
    row: Mapping[str, object],
    codecs: CodecBundle | tuple[TokenCodec, ActionCodec] | Mapping[str, object],
    device: str | torch.device,
    action_budget: int,
) -> DecodeResult:
    """Greedily decode actions from inventory/goal tokens with stateful legal masking."""
    bundle = _unpack_codecs(codecs)
    device = torch.device(device)
    graph = CoverageGraph(
        int(row["N"]),
        [tuple(edge) for edge in row.get("graph_edges", [])],
    )
    goal = tree_from_dict(row["goal"])
    env = CoverageEnv(graph)
    inventory = tuple(tree_from_dict(item) for item in row["inventory"])
    state = EnvState(inventory)
    source_tokens = encode_source(row, bundle.token)
    src = torch.tensor(source_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_padding_mask = src.eq(bundle.token.PAD)
    decoder_tokens = [bundle.action.bos_token]
    actions: list[MergeAction] = []
    legal_actions: list[tuple[MergeAction, ...]] = []
    emitted_tokens: list[int] = []

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max(0, action_budget)):
                decoder_input = torch.tensor(decoder_tokens, dtype=torch.long, device=device).unsqueeze(0)
                decoder_padding_mask = decoder_input.eq(bundle.action.pad_token)
                logits = model(src, decoder_input, src_padding_mask, decoder_padding_mask)
                if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != bundle.action.vocab_size:
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                current_legal = tuple(env.valid_actions(state))
                legal_actions.append(current_legal)
                mask = legal_action_mask(state, bundle.action).to(device)
                masked_logits = logits[0, -1].masked_fill(~mask, float("-inf"))
                if not torch.isfinite(masked_logits).any():
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                token = int(masked_logits.argmax().item())
                emitted_tokens.append(token)
                decoder_tokens.append(token)
                if token == bundle.action.eos_token:
                    checked = check_trace(graph, goal, actions, inventory=inventory)
                    if checked.valid:
                        return _result(True, actions, legal_actions, True, "none", emitted_tokens)
                    return _result(False, actions, legal_actions, False, "wrong_final_tree", emitted_tokens)
                try:
                    action = bundle.action.decode(token)
                except (TypeError, ValueError):
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                if action is None or action not in current_legal:
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                try:
                    state = env.apply(state, action)
                except (TypeError, ValueError):
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                actions.append(action)
    finally:
        model.train(model_was_training)
    return _result(False, actions, legal_actions, False, "budget_exhausted", emitted_tokens)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class SFTModel:
    backend: str
    episodes: int
    seed: int

    def predict_policy(self) -> str:
        return "random"


def train_sft_model(
    records: Iterable[dict[str, object]],
    output_path: str | Path | None = None,
    seed: int = 0,
) -> SFTModel:
    records = list(records)
    backend = "torch" if torch_available() else "fallback"
    model = SFTModel(backend=backend, episodes=len(records), seed=seed)
    if output_path is not None:
        write_json(
            output_path,
            {"backend": model.backend, "episodes": model.episodes, "seed": model.seed},
        )
    return model
