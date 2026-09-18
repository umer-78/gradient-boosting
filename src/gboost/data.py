"""Loading a CSV into features, a target and column names."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Table:
    x: np.ndarray
    y: np.ndarray
    features: list[str]
    target: str

    def __len__(self) -> int:
        return len(self.y)

    def split(self, fraction: float = 0.7, seed: int = 0) -> tuple[Table, Table]:
        """A shuffled split. These rows have no time order, so shuffling is right here —
        it would not be on a time series, where the split must be by date."""
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between 0 and 1")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self))
        cut = int(len(self) * fraction)
        if cut < 1 or len(self) - cut < 1:
            raise ValueError("not enough rows to split")
        first, second = order[:cut], order[cut:]
        return (Table(self.x[first], self.y[first], self.features, self.target),
                Table(self.x[second], self.y[second], self.features, self.target))


def read_csv(path: str | Path, target: str | None = None) -> Table:
    """Read a numeric CSV. The target is the last column unless named."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        raise ValueError(f"{path.name} holds no data rows")

    header = [h.strip() for h in rows[0]]
    target = target or header[-1]
    if target not in header:
        raise ValueError(f"no column named {target!r} in {', '.join(header)}")

    index = header.index(target)
    values = np.empty((len(rows) - 1, len(header)), dtype=np.float64)
    for line, row in enumerate(rows[1:], start=2):
        try:
            values[line - 2] = [float(v) for v in row]
        except ValueError as error:
            raise ValueError(f"{path.name} line {line}: {error}") from None

    features = [h for i, h in enumerate(header) if i != index]
    return Table(np.delete(values, index, axis=1), values[:, index], features, target)
