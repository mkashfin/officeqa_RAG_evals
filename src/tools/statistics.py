"""Statistical tools for OfficeQA-style computation questions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def trimmed_mean(values: Iterable[float], trim_fraction: float = 0.2) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("No values for trimmed mean.")
    arr.sort()
    trim_count = int(math.floor(arr.size * trim_fraction))
    if trim_count * 2 >= arr.size:
        trimmed = arr
    else:
        trimmed = arr[trim_count : arr.size - trim_count]
    return float(np.mean(trimmed))


def tukey_quartile_q1(values: Iterable[float]) -> float:
    """Tukey exclusive-method Q1: median of the lower half of the data."""
    arr = np.sort(np.asarray(list(values), dtype=float))
    if arr.size == 0:
        raise ValueError("No values for quartile.")
    lower = arr[: arr.size // 2]
    return float(np.median(lower))


def hp_filter(series: Iterable[float], lamb: float = 100.0) -> tuple[list[float], list[float]]:
    """Hodrick-Prescott filter returning trend and cycle components."""
    y = np.asarray(list(series), dtype=float)
    n = y.size
    if n < 4:
        raise ValueError("HP filter requires at least 4 observations.")

    e = np.eye(n)
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i] = 1
        d2[i, i + 1] = -2
        d2[i, i + 2] = 1

    trend = np.linalg.solve(e + lamb * d2.T @ d2, y)
    cycle = y - trend
    return trend.tolist(), cycle.tolist()
