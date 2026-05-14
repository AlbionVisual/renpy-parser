from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _group(col: str) -> str:
    if col.startswith("jamendo_"):
        return "jamendo"
    if col.startswith("mfcc_"):
        return "mfcc"
    if col.startswith("spectral_"):
        return "spectral"
    if col.startswith("chroma"):
        return "chroma"
    return "other"


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * q))
    return xs[max(0, min(k, len(xs) - 1))]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Сводка per_column_metrics.csv: квантили и силу по префиксам групп столбцов.",
    )
    p.add_argument("csv_path", type=Path)
    args = p.parse_args()
    rows = list(csv.DictReader(args.csv_path.open(newline="", encoding="utf-8")))
    pearsons = [float(r["pearson"]) for r in rows]
    maes = [float(r["mae"]) for r in rows]
    print("file", str(args.csv_path))
    print("n_columns", len(rows))
    print("pearson p10 p50 p90", _pct(pearsons, 0.1), _pct(pearsons, 0.5), _pct(pearsons, 0.9))
    print("mae p10 p50 p90", _pct(maes, 0.1), _pct(maes, 0.5), _pct(maes, 0.9))
    for t in (0.0, 0.05, 0.08, 0.1, 0.15, 0.2):
        c = sum(1 for x in pearsons if x >= t)
        print("count pearson>=", t, c)
    by_g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_g[_group(r["y_column"])].append(abs(float(r["pearson"])))
    print("by_prefix median_abs_pearson mean_abs_pearson n")
    for g in sorted(by_g.keys()):
        xs = by_g[g]
        med = statistics.median(xs)
        mean = sum(xs) / len(xs)
        print(g, round(med, 4), round(mean, 4), len(xs))


if __name__ == "__main__":
    main()
