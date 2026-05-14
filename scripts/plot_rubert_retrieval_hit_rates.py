from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    plt = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bar chart of retrieval cosine hit rate from metrics_summary.csv (main-protocol filter applied upstream).",
    )
    p.add_argument("--csv", type=Path, default=Path("artifacts/summary.csv"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("coursework-latex/lyrata-kurs/images/retrieval_cosine_hit_p35.png"),
    )
    p.add_argument("--top", type=int, default=14)
    args = p.parse_args()
    if plt is None:
        print("matplotlib required:", _IMPORT_ERR)
        return
    if not args.csv.is_file():
        print("csv missing", args.csv)
        return
    rows: list[dict[str, str]] = []
    with args.csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    for row in rows:
        row["_key"] = float(row.get("retrieval_cosine_hit_rate") or 0.0)
    rows.sort(key=lambda x: x["_key"], reverse=True)
    rows = rows[: max(1, int(args.top))]
    labels = [r.get("run_id", "") for r in rows]
    vals = [float(r.get("retrieval_cosine_hit_rate") or 0.0) for r in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.barh(labels[::-1], vals[::-1], color="steelblue")
    ax.axvline(0.35, color="crimson", linestyle="--", linewidth=1.0, label="baseline ~ p/100")
    ax.set_xlabel("retrieval cosine hit rate (test)")
    ax.set_title("RuBERT→music: leave_games_out, top 35% candidates, ho_seed=17")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.0, max(0.85, max(vals) * 1.05))
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
