"""Safe calculator and financial math helpers."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Safely evaluate a numeric expression."""
    tree = ast.parse(expression.strip(), mode="eval")
    return float(_eval_node(tree.body))


def percent_change(old: float, new: float, absolute: bool = False) -> float:
    if old == 0:
        raise ValueError("Cannot compute percent change from zero baseline.")
    value = ((new - old) / abs(old)) * 100.0
    return abs(value) if absolute else value


def round_to(value: float, decimals: int = 0) -> float | int:
    rounded = round(value, decimals)
    if decimals == 0:
        return int(rounded)
    return rounded


def compound_annual_growth_rate(start: float, end: float, periods: int) -> float:
    if start <= 0 or periods <= 0:
        raise ValueError("CAGR requires positive start value and periods.")
    return ((end / start) ** (1 / periods) - 1) * 100.0


def project_cagr(base: float, rate_percent: float, periods: int) -> float:
    return base * ((1 + rate_percent / 100.0) ** periods)


def percent_difference_relative_to_actual(projected: float, actual: float) -> float:
    if actual == 0:
        raise ValueError("Actual value must be non-zero.")
    return ((projected - actual) / actual) * 100.0
