from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Iterable, Union


@dataclass(frozen=True)
class Node:
    left: "Tree"
    right: "Tree"

    def leaves(self) -> tuple[int, ...]:
        return tree_leaves(self)


Tree = Union[int, Node]


def tree_leaves(tree: Tree) -> tuple[int, ...]:
    if isinstance(tree, int):
        return (tree,)
    return tree_leaves(tree.left) + tree_leaves(tree.right)


def tree_depth(tree: Tree) -> int:
    if isinstance(tree, int):
        return 0
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


def tree_subtrees(tree: Tree) -> tuple[Tree, ...]:
    if isinstance(tree, int):
        return (tree,)
    return (tree,) + tree_subtrees(tree.left) + tree_subtrees(tree.right)


def tree_to_dict(tree: Tree) -> int | dict[str, object]:
    if isinstance(tree, int):
        return tree
    return {"left": tree_to_dict(tree.left), "right": tree_to_dict(tree.right)}


def tree_from_dict(value: int | dict[str, object]) -> Tree:
    if isinstance(value, int):
        return value
    return Node(tree_from_dict(value["left"]), tree_from_dict(value["right"]))


def _minimum_depth(n_leaves: int) -> int:
    depth = 0
    capacity = 1
    while capacity < n_leaves:
        capacity *= 2
        depth += 1
    return depth


def make_goal(
    n_leaves: int,
    requested_depth: int,
    seed: int = 0,
    atom_ids: Iterable[int] | None = None,
) -> Node:
    if n_leaves < 2:
        raise ValueError("n_leaves must be at least 2")
    if requested_depth < 1:
        raise ValueError("requested_depth must be positive")
    labels = tuple(range(n_leaves)) if atom_ids is None else tuple(atom_ids)
    if len(labels) != n_leaves or len(set(labels)) != n_leaves:
        raise ValueError("atom_ids must contain n_leaves unique atom ids")
    actual_depth = max(_minimum_depth(n_leaves), min(requested_depth, n_leaves - 1))
    rng = Random(seed)

    def build(start: int, count: int, depth: int) -> Tree:
        if count == 1:
            return labels[start]
        candidates: list[tuple[int, int, int]] = []
        for split in range(1, count):
            left_count = split
            right_count = count - split
            left_depth = min(depth - 1, left_count - 1)
            right_depth = min(depth - 1, right_count - 1)
            if _minimum_depth(left_count) > left_depth:
                continue
            if _minimum_depth(right_count) > right_depth:
                continue
            if max(left_depth, right_depth) != depth - 1:
                continue
            candidates.append((split, left_depth, right_depth))
        if not candidates:
            raise ValueError(f"cannot build {count} leaves at depth {depth}")
        split, left_depth, right_depth = rng.choice(candidates)
        return Node(
            build(start, split, left_depth),
            build(start + split, count - split, right_depth),
        )

    result = build(0, n_leaves, actual_depth)
    if not isinstance(result, Node):
        raise AssertionError("a multi-leaf goal must be a Node")
    return result


@dataclass(frozen=True)
class CoverageGraph:
    n: int
    edges: frozenset[tuple[int, int]]

    def __init__(self, n: int, edges: Iterable[tuple[int, int]]):
        if n < 1:
            raise ValueError("n must be positive")
        normalized: set[tuple[int, int]] = set()
        for left, right in edges:
            if left == right or not (0 <= left < n and 0 <= right < n):
                raise ValueError("edges must be simple and use valid atom ids")
            normalized.add(tuple(sorted((left, right))))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "edges", frozenset(normalized))

    @classmethod
    def random(cls, n: int, lam: float, seed: int) -> "CoverageGraph":
        if lam < 0:
            raise ValueError("lambda must be non-negative")
        m = round(lam * n / 2)
        max_edges = n * (n - 1) // 2
        m = min(m, max_edges)
        rng = Random(seed)
        all_edges = list(combinations(range(n), 2))
        return cls(n, rng.sample(all_edges, m))

    @classmethod
    def complete(cls, n: int) -> "CoverageGraph":
        return cls(n, combinations(range(n), 2))

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def rho(self) -> float:
        if self.n < 2:
            return 0.0
        return 2.0 * self.m / (self.n * (self.n - 1))

    def component_labels(self) -> tuple[int, ...]:
        labels = [-1] * self.n
        component = 0
        adjacency = [[] for _ in range(self.n)]
        for left, right in self.edges:
            adjacency[left].append(right)
            adjacency[right].append(left)
        for atom in range(self.n):
            if labels[atom] != -1:
                continue
            stack = [atom]
            labels[atom] = component
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = component
                        stack.append(neighbor)
            component += 1
        return tuple(labels)

    def same_component(self, left_atoms: Iterable[int], right_atoms: Iterable[int]) -> bool:
        labels = self.component_labels()
        left_labels = {labels[atom] for atom in left_atoms}
        right_labels = {labels[atom] for atom in right_atoms}
        return bool(left_labels & right_labels)

    def manifest(self, lam: float, seed: int) -> dict[str, object]:
        return {
            "lambda": float(lam),
            "graph_seed": int(seed),
            "N": self.n,
            "m": self.m,
            "rho": self.rho,
            "edges": [list(edge) for edge in sorted(self.edges)],
            "component_labels": list(self.component_labels()),
        }
