"""Writes the sample datasets in data/.

Both carry a deliberate trap: a column of pure noise with a different value in
every row. It is there so the difference between gain importance and permutation
importance is visible from the command line, not only in a paragraph.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 20260918
DATA = Path(__file__).resolve().parent.parent / "data"


def housing(n: int = 3000) -> tuple[list[str], np.ndarray]:
    """A price-like target: non-linear, with interactions and a noise column."""
    rng = np.random.default_rng(SEED)

    rooms = rng.integers(1, 7, n).astype(float)
    area = np.clip(rng.normal(95, 35, n), 25, 400)
    age = rng.integers(0, 80, n).astype(float)
    distance = np.clip(rng.exponential(8, n), 0.2, 45)
    quality = rng.integers(1, 6, n).astype(float)
    reference = rng.normal(size=n)                       # pure noise, all distinct

    price = (
        45 * area
        + 900 * rooms ** 1.4
        - 120 * age
        - 380 * distance
        + 2600 * quality
        + 40 * area * (quality >= 4)                     # an interaction
        + rng.normal(0, 1400, n)
    )

    columns = ["rooms", "area_m2", "age_years", "km_to_centre", "quality", "reference_id", "price"]
    return columns, np.column_stack([rooms, area, age, distance, quality, reference, price])


def churn(n: int = 4000) -> tuple[list[str], np.ndarray]:
    """A binary target with a threshold effect and the same noise column."""
    rng = np.random.default_rng(SEED + 1)

    tenure = rng.integers(0, 72, n).astype(float)
    monthly = np.clip(rng.normal(68, 25, n), 15, 160)
    tickets = rng.poisson(1.1, n).astype(float)
    contract = rng.integers(0, 3, n).astype(float)       # 0 monthly, 1 yearly, 2 two-year
    autopay = rng.integers(0, 2, n).astype(float)
    reference = rng.normal(size=n)

    logit = (
        1.8
        - 0.055 * tenure
        + 0.016 * monthly
        + 0.42 * tickets
        - 1.35 * contract
        - 0.65 * autopay
        + 0.9 * (tenure < 6)                             # the first months matter most
    )
    probability = 1 / (1 + np.exp(-logit))
    label = (rng.random(n) < probability).astype(float)

    columns = ["tenure_months", "monthly_charge", "support_tickets", "contract",
               "autopay", "reference_id", "churned"]
    return columns, np.column_stack([tenure, monthly, tickets, contract, autopay, reference, label])


def write(name: str, columns: list[str], rows: np.ndarray) -> None:
    path = DATA / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{v:.4f}".rstrip("0").rstrip(".") for v in row])
    print(f"{name}: {len(rows)} rows, {len(columns)} columns")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    write("housing.csv", *housing())
    write("churn.csv", *churn())


if __name__ == "__main__":
    main()
