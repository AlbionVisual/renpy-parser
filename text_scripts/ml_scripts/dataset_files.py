from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import psycopg

from text_scripts.ml_scripts.train_rubert_to_music_linear import (
    LoadedFrames,
    PairRow,
    default_dsn,
    retrieval_scores,
)


def training_db_context(args: argparse.Namespace):
    if getattr(args, "dataset_dir", None) is not None:
        return nullcontext(None)
    dsn = args.dsn or default_dsn()
    return psycopg.connect(dsn)


def apply_merge_identical_embeddings_if_requested(
    loaded: LoadedFrames,
    merge: bool,
) -> LoadedFrames:
    if not merge:
        return loaded
    from text_scripts.ml_scripts.train_rubert_to_music_linear import (
        merge_consecutive_identical_embeddings,
        rows_to_matrices,
    )

    rows_full = [
        PairRow(
            game=r.game,
            phrase_order=r.phrase_order,
            music_id=r.music_id,
            path=r.path,
            x=np.asarray(loaded.x[i], dtype=np.float32, copy=False),
            y=np.asarray(loaded.y[i], dtype=np.float32, copy=False),
        )
        for i, r in enumerate(loaded.rows)
    ]
    merged = merge_consecutive_identical_embeddings(rows_full)
    if len(merged) == len(loaded.rows):
        return loaded
    x, y = rows_to_matrices(merged)
    return LoadedFrames(y_columns=list(loaded.y_columns), rows=merged, x=x, y=y)


def load_dataset_dir(path: Path | str) -> LoadedFrames:
    root = Path(path).expanduser().resolve()
    x_path = root / "x.npy"
    y_path = root / "y.npy"
    yc_path = root / "y_columns.json"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("dataset needs x.npy and y.npy in " + str(root))
    x = np.load(x_path)
    y = np.load(y_path)
    y_columns: list[str] = json.loads(yc_path.read_text(encoding="utf-8"))
    meta_csv = root / "meta.csv"
    meta_pq = root / "meta.parquet"
    if meta_csv.exists():
        rows = _rows_from_meta_csv(meta_csv, int(x.shape[0]))
    elif meta_pq.exists():
        rows = _rows_from_meta_parquet(meta_pq, int(x.shape[0]))
    else:
        raise FileNotFoundError("dataset needs meta.csv or meta.parquet in " + str(root))
    if len(rows) != int(x.shape[0]) or len(rows) != int(y.shape[0]):
        raise ValueError("meta row count must match x and y")
    if int(y.shape[1]) != len(y_columns):
        raise ValueError("y dim must match y_columns length")
    prev: tuple[str, int, int] | None = None
    for i, r in enumerate(rows):
        key = (r.game, r.phrase_order, r.music_id)
        if prev is not None and key < prev:
            raise ValueError("meta rows must be sorted by (game, phrase_order, music_id)")
        prev = key
    return LoadedFrames(y_columns=y_columns, rows=rows, x=x.astype(np.float32, copy=False), y=y.astype(np.float32, copy=False))


def _rows_from_meta_csv(meta_csv: Path, n_expected: int) -> list[PairRow]:
    rows: list[PairRow] = []
    with meta_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            row_id = int(line["row_id"])
            if row_id != len(rows):
                raise ValueError("row_id must be 0..N-1 in order")
            game = str(line["game"])
            phrase_order = int(line["phrase_order"])
            music_id = int(line["music_id"])
            p = line.get("path") or ""
            path = None if p == "" else str(p)
            rows.append(
                PairRow(
                    game=game,
                    phrase_order=phrase_order,
                    music_id=music_id,
                    path=path,
                    x=np.zeros(0, dtype=np.float32),
                    y=np.zeros(0, dtype=np.float32),
                )
            )
    if len(rows) != n_expected:
        raise ValueError("meta.csv row count mismatch")
    return rows


def _rows_from_meta_parquet(meta_pq: Path, n_expected: int) -> list[PairRow]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("reading meta.parquet requires pyarrow") from e
    t = pq.read_table(meta_pq)
    n = int(t.num_rows)
    if n != n_expected:
        raise ValueError("meta.parquet row count mismatch")
    games = t.column("game").to_pylist()
    phrase_orders = t.column("phrase_order").to_pylist()
    music_ids = t.column("music_id").to_pylist()
    paths = t.column("path").to_pylist() if "path" in t.column_names else [None] * n
    row_ids = t.column("row_id").to_pylist() if "row_id" in t.column_names else list(range(n))
    rows: list[PairRow] = []
    for i in range(n):
        if int(row_ids[i]) != i:
            raise ValueError("row_id must be 0..N-1 in order")
        p = paths[i]
        path = None if p is None or str(p) == "" else str(p)
        rows.append(
            PairRow(
                game=str(games[i]),
                phrase_order=int(phrase_orders[i]),
                music_id=int(music_ids[i]),
                path=path,
                x=np.zeros(0, dtype=np.float32),
                y=np.zeros(0, dtype=np.float32),
            )
        )
    return rows


def load_candidates_dir(dataset_dir: Path | str) -> dict[str, list[tuple[int, np.ndarray]]] | None:
    root = Path(dataset_dir).expanduser().resolve()
    idx_path = root / "candidates" / "index.json"
    if not idx_path.exists():
        return None
    index: dict[str, str] = json.loads(idx_path.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for game, fn in index.items():
        p = root / "candidates" / fn
        z = np.load(p)
        ids = z["music_id"]
        mat = z["y"]
        for j in range(int(ids.shape[0])):
            out[str(game)].append((int(ids[j]), mat[j].astype(np.float32, copy=False)))
    return dict(out)


def retrieval_hit_rates_from_candidates(
    candidates_by_game: dict[str, list[tuple[int, np.ndarray]]],
    test_rows: Sequence[PairRow],
    y_pred: np.ndarray,
    top_percent: float,
    column_indices: np.ndarray | None = None,
) -> dict[str, tuple[int, int, int]]:
    empty = {m: (0, 0, 0) for m in ("rmse", "mae", "cosine")}
    try:
        hits = {"rmse": 0, "mae": 0, "cosine": 0}
        totals = {"rmse": 0, "mae": 0, "cosine": 0}
        skipped = 0
        for i, row in enumerate(test_rows):
            candidates = candidates_by_game.get(row.game, [])
            if len(candidates) <= 1:
                skipped += 1
                continue
            ids = [music_id for music_id, _vec in candidates]
            if row.music_id not in ids:
                skipped += 1
                continue
            mat = np.stack([vec for _music_id, vec in candidates], axis=0)
            top_n = max(1, int(math.ceil(len(ids) * top_percent / 100.0)))
            yp = y_pred[i]
            for metric in ("rmse", "mae", "cosine"):
                scores = retrieval_scores(yp, mat, metric, column_indices)
                if metric == "cosine":
                    order = np.argsort(-scores)
                else:
                    order = np.argsort(scores)
                top_ids = {ids[j] for j in order[:top_n]}
                hits[metric] += int(row.music_id in top_ids)
                totals[metric] += 1
        return {
            metric: (hits[metric], totals[metric], skipped)
            for metric in ("rmse", "mae", "cosine")
        }
    except Exception as exc:
        print(
            "retrieval_hit_rates_from_candidates failed",
            type(exc).__name__,
            str(exc),
            "(returning empty retrieval counts)",
        )
        traceback.print_exc()
        return empty
