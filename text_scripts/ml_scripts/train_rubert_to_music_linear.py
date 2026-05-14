from __future__ import annotations
import argparse
import csv
import json
import math
import os
import secrets
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import explained_variance_score
from sklearn.multioutput import MultiOutputRegressor

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from text_scripts.ml_scripts.db_consts import TEXT_MUSICS_RUBERT_TABLE
from text_scripts.ml_scripts.retrieval_column_mask import (
    register_retrieval_column_mask_cli,
)
from text_scripts.ml_scripts.rubert_embeddings import validate_pg_identifier


YGroup = Literal["librosa", "jamendo", "all"]
SplitMode = Literal["within_game", "leave_games_out"]
ModelKind = Literal["ols", "ridge", "lasso"]

NUMERIC_Y_TYPES = {"double precision", "real", "numeric"}
DSP_PREFIXES = (
    "rms_",
    "spectral_",
    "mfcc_",
    "chroma_",
    "climax_",
    "energy_change_",
)
DSP_EXACT_NAMES = {
    "duration_sec",
    "dyn_range_p95_p05",
    "tempo",
    "onset_rate_hz",
    "beat_strength_mean",
    "beat_strength_std",
    "chroma_entropy",
    "chroma_peakiness",
}


@dataclass(frozen=True)
class PairRow:
    game: str
    phrase_order: int
    music_id: int
    path: str | None
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class LoadedFrames:
    y_columns: list[str]
    rows: list[PairRow]
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SweepCombo:
    model: ModelKind
    alpha: float | None


def music_mode_note(mode: str) -> None:
    print(
        "music_mode",
        mode,
        "is accepted as a V1 stub: each row has one music id, so training is unchanged",
    )


def default_dsn() -> str:
    load_dotenv(_repo_root / ".env")
    load_dotenv()
    return os.environ["pghost"]


def is_dsp_column(name: str) -> bool:
    if name in DSP_EXACT_NAMES:
        return True
    return name.startswith(DSP_PREFIXES)


def select_y_columns(columns: Sequence[tuple[str, str]], y_group: YGroup) -> list[str]:
    out: list[str] = []
    for name, data_type in columns:
        if data_type not in NUMERIC_Y_TYPES:
            continue
        if name in ("id", "game", "path"):
            continue
        is_jamendo = name.startswith("jamendo_")
        is_librosa = is_dsp_column(name)
        if y_group == "jamendo" and is_jamendo:
            out.append(name)
        elif y_group == "librosa" and is_librosa and not is_jamendo:
            out.append(name)
        elif y_group == "all" and (is_jamendo or is_librosa):
            out.append(name)
    return out


def fetch_music_columns(conn: psycopg.Connection, music_table: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select column_name, data_type
            from information_schema.columns
            where table_schema = current_schema()
              and table_name = %s
            order by ordinal_position
            """,
            (music_table,),
        )
        return [(str(name), str(data_type)) for name, data_type in cur.fetchall()]


def not_null_expr(table_alias: str, columns: Sequence[str]) -> sql.SQL:
    parts = [
        sql.SQL(".").join([sql.Identifier(table_alias), sql.Identifier(c)])
        + sql.SQL(" is not null")
        for c in columns
    ]
    return sql.SQL(" and ").join(parts)


def as_float32_vector(value: object, expected_dim: int | None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("embedding must be a 1D vector")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError("embedding dimension changed")
    return arr


def fetch_pairs(
    conn: psycopg.Connection,
    rubert_table: str,
    embedding_column: str,
    music_table: str,
    y_columns: Sequence[str],
) -> list[PairRow]:
    y_select = sql.SQL(", ").join(
        sql.SQL(".").join([sql.Identifier("md"), sql.Identifier(c)]) for c in y_columns
    )
    emb_col = validate_pg_identifier(embedding_column)
    query = sql.SQL(
        """
        select
            t.game,
            t.phrase_order,
            t.music,
            md.game,
            md.path,
            t.{emb_col},
            {y_select}
        from {rubert_table} t
        join {music_table} md on t.music = md.id
        where t.{emb_col} is not null
          and t.music is not null
          and {y_not_null}
        order by md.game, t.phrase_order
        """
    ).format(
        emb_col=sql.Identifier(emb_col),
        y_select=y_select,
        rubert_table=sql.Identifier(rubert_table),
        music_table=sql.Identifier(music_table),
        y_not_null=not_null_expr("md", y_columns),
    )

    rows: list[PairRow] = []
    x_dim: int | None = None
    with conn.cursor() as cur:
        cur.execute(query)
        for row in cur.fetchall():
            _text_game, phrase_order, music_id, music_game, path, emb, *y_values = row
            x = as_float32_vector(emb, x_dim)
            if x_dim is None:
                x_dim = int(x.shape[0])
            y = np.asarray(y_values, dtype=np.float32)
            rows.append(
                PairRow(
                    game=str(music_game),
                    phrase_order=int(phrase_order),
                    music_id=int(music_id),
                    path=None if path is None else str(path),
                    x=x,
                    y=y,
                ),
            )
    return rows


def rows_to_matrices(rows: Sequence[PairRow]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([r.x for r in rows], axis=0).astype(np.float32, copy=False)
    y = np.stack([r.y for r in rows], axis=0).astype(np.float32, copy=False)
    return x, y


def merge_consecutive_identical_embeddings(rows: Sequence[PairRow]) -> list[PairRow]:
    if not rows:
        return []
    out: list[PairRow] = []
    cur = rows[0]
    run_len = 1
    for r in rows[1:]:
        if r.game == cur.game and r.music_id == cur.music_id and np.array_equal(r.x, cur.x):
            run_len += 1
            continue
        out.append(cur)
        cur = r
        run_len = 1
    out.append(cur)
    return out


def split_within_game(
    rows: Sequence[PairRow],
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_game: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_game[row.game].append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for indices in by_game.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_idx.extend(shuffled)
            continue
        n_test = int(math.ceil(len(shuffled) * test_fraction))
        n_test = max(1, min(n_test, len(shuffled) - 1))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])
    return train_idx, test_idx


def split_leave_games_out(
    rows: Sequence[PairRow],
    test_games: int,
    games_shuffle_seed: int,
    *,
    test_game_min_rows: int | None = None,
    test_game_max_rows: int | None = None,
) -> tuple[list[int], list[int], list[str]]:
    games = sorted({r.game for r in rows})
    if len(games) < 2:
        raise ValueError("leave_games_out needs at least 2 games")
    n_test_games = max(1, min(test_games, len(games) - 1))
    rng = np.random.default_rng(int(games_shuffle_seed))
    counts = Counter(r.game for r in rows)
    if test_game_min_rows is not None and test_game_max_rows is not None:
        pool = [
            g
            for g in games
            if int(test_game_min_rows) <= counts[g] <= int(test_game_max_rows)
        ]
        print(
            "leave_games_out test pool by row count",
            "min_rows",
            int(test_game_min_rows),
            "max_rows",
            int(test_game_max_rows),
            "eligible_games",
            len(pool),
            "need_test_games",
            n_test_games,
        )
        if len(pool) < n_test_games:
            raise ValueError(
                "not enough games in row band for test: need "
                + str(n_test_games)
                + " games with row count in ["
                + str(int(test_game_min_rows))
                + ", "
                + str(int(test_game_max_rows))
                + "], found "
                + str(len(pool))
            )
        shuffled = list(pool)
        rng.shuffle(shuffled)
        test_game_set = set(shuffled[:n_test_games])
    else:
        shuffled = list(games)
        rng.shuffle(shuffled)
        test_game_set = set(shuffled[:n_test_games])
    train_idx = [i for i, r in enumerate(rows) if r.game not in test_game_set]
    test_idx = [i for i, r in enumerate(rows) if r.game in test_game_set]
    return train_idx, test_idx, sorted(test_game_set)


def validate_held_out_game_row_bounds(args: argparse.Namespace) -> int | None:
    lo = getattr(args, "held_out_game_min_rows", None)
    hi = getattr(args, "held_out_game_max_rows", None)
    if lo is None and hi is None:
        return None
    if getattr(args, "split", None) != "leave_games_out":
        print(
            "--held-out-game-min-rows / --held-out-game-max-rows only apply with --split leave_games_out",
        )
        return 2
    if lo is None or hi is None:
        print(
            "set both --held-out-game-min-rows and --held-out-game-max-rows or omit both",
        )
        return 2
    if int(lo) < 1 or int(hi) < 1:
        print("held-out game row bounds must be >= 1")
        return 2
    if int(lo) > int(hi):
        print("--held-out-game-min-rows must be <= --held-out-game-max-rows")
        return 2
    return None


def row_bounds_kwargs_for_leave_games_out(args: argparse.Namespace) -> dict[str, int]:
    lo = getattr(args, "held_out_game_min_rows", None)
    hi = getattr(args, "held_out_game_max_rows", None)
    if lo is None:
        return {}
    return {"test_game_min_rows": int(lo), "test_game_max_rows": int(hi)}


def split_leave_games_out_from_args(
    rows: Sequence[PairRow],
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[str]] | None:
    try:
        return split_leave_games_out(
            rows,
            args.test_games,
            int(args.held_out_games_seed),
            **row_bounds_kwargs_for_leave_games_out(args),
        )
    except ValueError as e:
        print("leave_games_out:", e)
        return None


def build_model(kind: ModelKind, alpha: float, *, rng_seed: int) -> LinearRegression | Ridge | MultiOutputRegressor:
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=float(alpha))
    return MultiOutputRegressor(
        Lasso(
            alpha=float(alpha),
            max_iter=25000,
            tol=1e-4,
            random_state=int(rng_seed),
        ),
    )


def parse_models(value: str) -> list[ModelKind]:
    models: list[ModelKind] = []
    for raw in value.split(","):
        model = raw.strip()
        if not model:
            continue
        if model not in ("ols", "ridge", "lasso"):
            raise argparse.ArgumentTypeError(
                "--sweep-models can contain only ols,ridge,lasso")
        models.append(model)
    if not models:
        raise argparse.ArgumentTypeError("--sweep-models is empty")
    return models


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            out.append(float(raw))
    if not out:
        raise argparse.ArgumentTypeError("float list is empty")
    return out


def sweep_combos(
    models: Sequence[ModelKind],
    ridge_alphas: Sequence[float],
    lasso_alphas: Sequence[float],
) -> list[SweepCombo]:
    out: list[SweepCombo] = []
    for model in models:
        if model == "ols":
            out.append(SweepCombo(model=model, alpha=None))
        elif model == "ridge":
            for alpha in ridge_alphas:
                out.append(SweepCombo(model=model, alpha=float(alpha)))
        else:
            for alpha in lasso_alphas:
                out.append(SweepCombo(model=model, alpha=float(alpha)))
    return out


def split_validation_from_train(
    train_idx: Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        return list(train_idx), []
    if len(train_idx) < 3:
        return list(train_idx), []
    shuffled = list(train_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    n_val = int(math.ceil(len(shuffled) * val_fraction))
    n_val = max(1, min(n_val, len(shuffled) - 1))
    return shuffled[n_val:], shuffled[:n_val]


def split_validation_games_held_out_from_train(
    rows: Sequence[PairRow],
    train_idx: Sequence[int],
    val_games: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    games = sorted({rows[i].game for i in train_idx})
    if len(games) < 2:
        return list(train_idx), []
    n_take = min(max(1, int(val_games)), len(games) - 1)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(games, dtype=object),
                        size=n_take, replace=False)
    val_game_set = set(chosen.tolist())
    tune_train_idx = [i for i in train_idx if rows[i].game not in val_game_set]
    val_idx = [i for i in train_idx if rows[i].game in val_game_set]
    if not tune_train_idx or not val_idx:
        return list(train_idx), []
    return tune_train_idx, val_idx


def split_validation_tail_per_game_from_train(
    rows: Sequence[PairRow],
    train_idx: Sequence[int],
    val_fraction: float,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        return list(train_idx), []
    by_game: dict[str, list[int]] = defaultdict(list)
    for i in train_idx:
        by_game[rows[i].game].append(i)
    tune_train_idx: list[int] = []
    val_idx: list[int] = []
    for game in sorted(by_game.keys()):
        idxs = sorted(
            by_game[game],
            key=lambda i: (rows[i].phrase_order, rows[i].music_id),
        )
        m = len(idxs)
        if m < 2:
            tune_train_idx.extend(idxs)
            continue
        n_val = int(math.ceil(m * val_fraction))
        n_val = max(1, min(n_val, m - 1))
        split_at = m - n_val
        tune_train_idx.extend(idxs[:split_at])
        val_idx.extend(idxs[split_at:])
    if not val_idx:
        return list(train_idx), []
    return tune_train_idx, val_idx


def game_count(rows: Sequence[PairRow], indices: Sequence[int]) -> int:
    return len({rows[i].game for i in indices})


def print_split_visibility(
    rows: Sequence[PairRow],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    args: argparse.Namespace,
    held_out_games: Sequence[str] | None,
) -> None:
    train_set = set(train_idx)
    test_set = set(test_idx)
    print("")
    print("TRAIN/TEST SPLIT BEFORE TRAINING")
    print("seed", args.seed)
    print("split mode", args.split)
    print("split test_fraction", args.test_fraction)
    print("split test_games requested", args.test_games)
    if held_out_games is not None:
        print("split held_out_games", len(held_out_games), list(held_out_games))
    print("train rows", len(train_idx),
          "games covered", game_count(rows, train_idx))
    print("test rows", len(test_idx), "games covered", game_count(rows, test_idx))
    print("train_test_index_overlap", len(train_set & test_set))
    print("tune_split val_fraction", args.tune_split)
    print("refit_best_on_train_plus_val", bool(
        args.refit_best_on_train_plus_val))


def summarize_from_arrays(
    mae: np.ndarray,
    rmse: np.ndarray,
    r2: np.ndarray,
    cosine: np.ndarray,
    medae: np.ndarray,
    evs: np.ndarray,
    max_abs_col: np.ndarray,
    pearson_col: np.ndarray,
    abs_p95_col: np.ndarray,
) -> dict[str, float]:
    return {
        "mae_macro": float(np.nanmean(mae)),
        "rmse_macro": float(np.nanmean(rmse)),
        "r2_macro": float(np.nanmean(r2)),
        "medae_macro": float(np.nanmean(medae)),
        "evs_macro": float(np.nanmean(evs)),
        "max_abs_macro": float(np.nanmean(max_abs_col)),
        "abs_p95_macro_mean": float(np.nanmean(abs_p95_col)),
        "mae_across_columns_p95": float(np.nanpercentile(mae, 95)),
        "column_pearson_mean": float(np.nanmean(pearson_col)),
        "column_pearson_median": float(np.nanmedian(pearson_col)),
        "column_pearson_across_columns_p95": float(np.nanpercentile(pearson_col, 95)),
        "column_pearson_nan_frac": float(np.mean(np.isnan(pearson_col))),
        "row_cosine_mean": float(np.nanmean(cosine)),
        "row_cosine_median": float(np.nanmedian(cosine)),
    }


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return summarize_from_arrays(*regression_metrics(y_true, y_pred))


def failed_regression_summary() -> dict[str, float]:
    x = float("nan")
    return {
        "mae_macro": x,
        "rmse_macro": x,
        "r2_macro": x,
        "medae_macro": x,
        "evs_macro": x,
        "max_abs_macro": x,
        "abs_p95_macro_mean": x,
        "mae_across_columns_p95": x,
        "column_pearson_mean": x,
        "column_pearson_median": x,
        "column_pearson_across_columns_p95": x,
        "column_pearson_nan_frac": x,
        "row_cosine_mean": x,
        "row_cosine_median": x,
        "r2_pooled": x,
    }


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_pred - y_true).astype(np.float64, copy=False)
    centered = y_true.astype(np.float64, copy=False) - np.mean(
        y_true.astype(np.float64, copy=False),
        axis=0,
        keepdims=True,
    )
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def print_regression_block(
    title: str,
    y_columns: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_column: bool,
) -> dict[str, float]:
    print("")
    print(title)
    try:
        mae, rmse, r2, cosine, medae, evs, max_abs_col, pearson_col, abs_p95_col = (
            regression_metrics(y_true, y_pred)
        )
        summary = summarize_from_arrays(
            mae,
            rmse,
            r2,
            cosine,
            medae,
            evs,
            max_abs_col,
            pearson_col,
            abs_p95_col,
        )
        r2_pool = r2_pooled(y_true, y_pred)
        summary["r2_pooled"] = float(r2_pool)
        print_metric_summary("mae", mae)
        print_metric_summary("rmse", rmse)
        print_metric_summary("r2", r2)
        print(
            "r2_pooled",
            round(float(r2_pool), 6),
            "note",
            "per-column r2 mean is mean of per-target R2; a few near-constant y columns (tiny ss_tot) make R2 explode negative; median and r2_pooled are safer headlines",
        )
        print_metric_summary("medae_column", medae)
        print_metric_summary("abs_error_p90_within_column", abs_p95_col)
        print_metric_summary("explained_variance", evs)
        print_metric_summary("max_abs_column", max_abs_col)
        print(
            "column_pearson mean",
            round(float(np.nanmean(pearson_col)), 6),
            "median",
            round(float(np.nanmedian(pearson_col)), 6),
            "p90_across_columns",
            round(float(np.nanpercentile(pearson_col, 90)), 6),
            "fraction_nan",
            round(float(np.mean(np.isnan(pearson_col))), 6),
        )
        print(
            "row_cosine mean",
            round(float(np.nanmean(cosine)), 6),
            "median",
            round(float(np.nanmedian(cosine)), 6),
            "p90_across_rows",
            round(float(np.nanpercentile(cosine, 90)), 6),
        )
        if per_column:
            print_per_column_extended(
                y_columns,
                mae,
                rmse,
                r2,
                medae,
                evs,
                max_abs_col,
                pearson_col,
                abs_p95_col,
            )
        if "TEST ONLY" in title:
            print_top_y_columns_by_pearson(
                y_columns,
                mae,
                medae,
                abs_p95_col,
                pearson_col,
                top_n=30,
            )
        return summary
    except Exception as exc:
        print(
            "REGRESSION_METRICS_FAILED",
            type(exc).__name__,
            str(exc),
            "(downstream steps continue; summary filled with NaN)",
        )
        traceback.print_exc()
        return failed_regression_summary()


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_cols = int(y_true.shape[1])
    row_ok = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    n_bad = int((~row_ok).sum())
    if n_bad:
        warnings.warn(
            "regression_metrics: dropping %d rows with non-finite y_true/y_pred" % n_bad,
            RuntimeWarning,
        )
    if not np.any(row_ok):
        warnings.warn(
            "regression_metrics: no finite rows; returning NaN column summaries",
            RuntimeWarning,
        )
        nanv = np.full(n_cols, np.nan, dtype=np.float64)
        zcos = np.zeros(0, dtype=np.float64)
        return nanv, nanv, nanv, zcos, nanv, nanv, nanv, nanv, nanv
    y_true = y_true[row_ok]
    y_pred = y_pred[row_ok]
    diff = y_pred - y_true
    abs_diff = np.abs(diff).astype(np.float64, copy=False)
    mae = np.mean(abs_diff, axis=0)
    medae = np.median(abs_diff, axis=0)
    abs_p95_col = np.nanpercentile(abs_diff, 95.0, axis=0)
    rmse = np.sqrt(np.mean(diff * diff, axis=0))
    ss_res = np.sum(diff * diff, axis=0)
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=0)
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    mask = ss_tot > 0
    r2[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    cosine = row_cosine(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred, multioutput="raw_values")
    max_abs_col = np.max(np.abs(diff), axis=0).astype(np.float64)
    pearson_col = column_pearson(y_true, y_pred)
    return mae, rmse, r2, cosine, medae, evs, max_abs_col, pearson_col, abs_p95_col


def column_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n_cols = y_true.shape[1]
    out = np.full(n_cols, np.nan, dtype=np.float64)
    eps = 1e-15
    for j in range(n_cols):
        a = y_true[:, j].astype(np.float64, copy=False)
        b = y_pred[:, j].astype(np.float64, copy=False)
        if np.std(a) < eps or np.std(b) < eps:
            continue
        out[j] = float(np.corrcoef(a, b)[0, 1])
    return out


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def print_metric_summary(name: str, values: np.ndarray) -> None:
    v = values.astype(np.float64, copy=False)
    print(
        name,
        "mean",
        round(float(np.nanmean(v)), 6),
        "median",
        round(float(np.nanmedian(v)), 6),
        "p90_across_columns",
        round(float(np.nanpercentile(v, 90.0)), 6),
    )


def print_top_y_columns_by_pearson(
    y_columns: Sequence[str],
    mae: np.ndarray,
    medae: np.ndarray,
    abs_p95_col: np.ndarray,
    pearson_col: np.ndarray,
    *,
    top_n: int,
) -> None:
    n = min(int(top_n), len(y_columns))
    order = sorted(
        range(len(y_columns)),
        key=lambda i: (
            float(pearson_col[i]) if not np.isnan(
                pearson_col[i]) else float("-inf"),
            -float(mae[i]),
        ),
        reverse=True,
    )[:n]
    print("")
    print(
        "top",
        n,
        "y_columns by pearson (tie-break lower mae); per-column medae / p90|err| / mae",
    )
    for j, i in enumerate(order, 1):
        pr = pearson_col[i]
        pr_s = round(float(pr), 6) if not np.isnan(pr) else "nan"
        print(
            j,
            y_columns[i],
            "pearson",
            pr_s,
            "mae",
            round(float(mae[i]), 6),
            "medae",
            round(float(medae[i]), 6),
            "abs_p90",
            round(float(abs_p95_col[i]), 6),
        )


def _csv_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)) and (math.isnan(float(v)) or math.isinf(float(v))):
        return ""
    if isinstance(v, (np.integer, np.floating)):
        return str(float(v))
    return str(v)


def write_per_column_metrics_csv(
    path: Path,
    *,
    y_columns: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: dict[str, str],
) -> None:
    try:
        mae, rmse, r2, _cosine, medae, evs, max_abs_col, pearson_col, abs_p95_col = (
            regression_metrics(
                y_true,
                y_pred,
            )
        )
        out = path.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        meta_keys = list(meta.keys())
        metric_keys = [
            "y_column",
            "mae",
            "medae",
            "abs_err_p90",
            "rmse",
            "r2",
            "explained_variance",
            "max_abs",
            "pearson",
        ]
        fieldnames = meta_keys + metric_keys
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i, col in enumerate(y_columns):
                row: dict[str, str] = {k: _csv_cell(meta[k]) for k in meta_keys}
                row["y_column"] = col
                row["mae"] = _csv_cell(float(mae[i]))
                row["medae"] = _csv_cell(float(medae[i]))
                row["abs_err_p90"] = _csv_cell(float(abs_p95_col[i]))
                row["rmse"] = _csv_cell(float(rmse[i]))
                row["r2"] = _csv_cell(float(r2[i]))
                row["explained_variance"] = _csv_cell(float(evs[i]))
                row["max_abs"] = _csv_cell(float(max_abs_col[i]))
                row["pearson"] = _csv_cell(float(pearson_col[i]))
                w.writerow(row)
    except Exception as exc:
        print(
            "write_per_column_metrics_csv failed",
            path,
            type(exc).__name__,
            str(exc),
        )
        traceback.print_exc()


def print_per_column_extended(
    y_columns: Sequence[str],
    mae: np.ndarray,
    rmse: np.ndarray,
    r2: np.ndarray,
    medae: np.ndarray,
    evs: np.ndarray,
    max_abs_col: np.ndarray,
    pearson_col: np.ndarray,
    abs_p95_col: np.ndarray,
) -> None:
    print("")
    print("per-column metrics")
    for i, col in enumerate(y_columns):
        pr = pearson_col[i]
        pr_s = round(float(pr), 6) if not np.isnan(pr) else "nan"
        print(
            col,
            "mae",
            round(float(mae[i]), 6),
            "medae",
            round(float(medae[i]), 6),
            "abs_err_p90",
            round(float(abs_p95_col[i]), 6),
            "rmse",
            round(float(rmse[i]), 6),
            "r2",
            round(float(r2[i]), 6) if not np.isnan(r2[i]) else "nan",
            "explained_variance",
            round(float(evs[i]), 6),
            "max_abs",
            round(float(max_abs_col[i]), 6),
            "pearson_col",
            pr_s,
        )


def print_imbalance(rows: Sequence[PairRow]) -> None:
    by_track = Counter(r.music_id for r in rows)
    tracks_by_id = {r.music_id: r for r in rows}
    print("")
    print("top tracks by row count")
    for music_id, count in by_track.most_common(20):
        row = tracks_by_id[music_id]
        print("music_id", music_id, "rows", count,
              "game", row.game, "path", row.path)

    tracks_per_game: dict[str, set[int]] = defaultdict(set)
    rows_per_game = Counter(r.game for r in rows)
    for row in rows:
        tracks_per_game[row.game].add(row.music_id)
    unique_counts = np.asarray(
        [len(v) for v in tracks_per_game.values()], dtype=np.float64)
    row_counts = np.asarray(list(rows_per_game.values()), dtype=np.float64)
    print("")
    print("game stats")
    print("games", len(tracks_per_game))
    print(
        "unique_tracks_per_game min",
        int(np.min(unique_counts)),
        "median",
        round(float(np.median(unique_counts)), 3),
        "max",
        int(np.max(unique_counts)),
    )
    print(
        "rows_per_game min",
        int(np.min(row_counts)),
        "median",
        round(float(np.median(row_counts)), 3),
        "max",
        int(np.max(row_counts)),
    )


def fetch_candidates(
    conn: psycopg.Connection,
    music_table: str,
    y_columns: Sequence[str],
    games: Sequence[str],
) -> dict[str, list[tuple[int, np.ndarray]]]:
    if not games:
        return {}
    y_select = sql.SQL(", ").join(
        sql.SQL(".").join([sql.Identifier("md"), sql.Identifier(c)]) for c in y_columns
    )
    query = sql.SQL(
        """
        select distinct md.id, md.game, {y_select}
        from {music_table} md
        where md.game = any(%s)
          and {y_not_null}
        order by md.game, md.id
        """
    ).format(
        y_select=y_select,
        music_table=sql.Identifier(music_table),
        y_not_null=not_null_expr("md", y_columns),
    )
    out: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(query, (list(games),))
        for row in cur.fetchall():
            music_id, game, *values = row
            out[str(game)].append(
                (int(music_id), np.asarray(values, dtype=np.float32)))
    return out


def load_frames(conn: psycopg.Connection, args: argparse.Namespace) -> LoadedFrames | None:
    columns = fetch_music_columns(conn, args.music_table)
    y_columns = select_y_columns(columns, args.y_group)
    if not y_columns:
        print("no Y columns selected for group", args.y_group)
        return None

    print("selected y columns", len(y_columns), "group", args.y_group)
    rows = fetch_pairs(conn, args.rubert_table,
                       args.embedding_column, args.music_table, y_columns)
    if len(rows) < 3:
        print("not enough rows after filtering", len(rows))
        return None
    if args.merge_identical_embeddings:
        merged_rows = merge_consecutive_identical_embeddings(rows)
        print(
            "merge_identical_embeddings",
            True,
            "rows_before",
            len(rows),
            "rows_after",
            len(merged_rows),
        )
        rows = merged_rows
    x, y = rows_to_matrices(rows)
    print("loaded rows", len(rows), "x_dim",
          x.shape[1], "y_dim", len(y_columns))
    print("rows with missing selected Y were dropped by SQL")
    print_imbalance(rows)
    return LoadedFrames(y_columns=y_columns, rows=rows, x=x, y=y)


def retrieval_scores(
    y_pred: np.ndarray,
    candidates: np.ndarray,
    metric: str,
    column_indices: np.ndarray | None = None,
) -> np.ndarray:
    yv = y_pred
    mat = candidates
    if column_indices is not None:
        yv = y_pred[column_indices]
        mat = candidates[:, column_indices]
    if metric == "rmse":
        diff = mat - yv[None, :]
        return np.sqrt(np.mean(diff * diff, axis=1))
    if metric == "mae":
        return np.mean(np.abs(mat - yv[None, :]), axis=1)
    num = mat @ yv
    den = np.linalg.norm(mat, axis=1) * np.linalg.norm(yv)
    out = np.full(candidates.shape[0], -np.inf, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def retrieval_hit_rates(
    conn: psycopg.Connection,
    music_table: str,
    y_columns: Sequence[str],
    test_rows: Sequence[PairRow],
    y_pred: np.ndarray,
    top_percent: float,
    column_indices: np.ndarray | None = None,
) -> dict[str, tuple[int, int, int]]:
    empty = {m: (0, 0, 0) for m in ("rmse", "mae", "cosine")}
    try:
        candidates_by_game = fetch_candidates(
            conn,
            music_table,
            y_columns,
            sorted({r.game for r in test_rows}),
        )
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
            for metric in ("rmse", "mae", "cosine"):
                scores = retrieval_scores(
                    y_pred[i], mat, metric, column_indices,
                )
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
            "retrieval_hit_rates failed",
            type(exc).__name__,
            str(exc),
            "(returning empty retrieval counts)",
        )
        traceback.print_exc()
        return empty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train linear baselines from text_musics_rubert embeddings column to music_data features.",
    )
    p.add_argument("--dsn", default=None,
                   help="psycopg DSN; defaults to env pghost")
    p.add_argument("--rubert-table", default=TEXT_MUSICS_RUBERT_TABLE)
    p.add_argument(
        "--embedding-column",
        default="text_analized",
        help="column in rubert table with embeddings real[] (e.g. text_analized_c12_s1_mean)",
    )
    p.add_argument(
        "--merge-identical-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge consecutive rows within the same game that have identical embedding vectors AND same music_id (default: on; --no-merge-identical-embeddings to disable)",
    )
    p.add_argument("--music-table", default="music_data")
    p.add_argument("--y-group", choices=("librosa",
                   "jamendo", "all"), default="all")
    p.add_argument("--split", choices=("within_game",
                   "leave_games_out"), default="within_game")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--test-games", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--held-out-games-seed",
        type=int,
        default=None,
        help="leave_games_out: сид для shuffle списка игр и выбора test-игр; "
        "по умолчанию случайный (значение печатается, задайте флаг с числом для воспроизводимости). "
        "Модель/данные по-прежнему --seed.",
    )
    p.add_argument(
        "--held-out-game-min-rows",
        type=int,
        default=None,
        help="leave_games_out: в test попадают только игры с числом строк в этом диапазоне (вместе с --held-out-game-max-rows)",
    )
    p.add_argument(
        "--held-out-game-max-rows",
        type=int,
        default=None,
        help="leave_games_out: верхняя граница числа строк у test-игр (вместе с --held-out-game-min-rows)",
    )
    p.add_argument(
        "--model",
        choices=("ols", "ridge", "lasso"),
        default="ridge",
        help="single-fit mode uses --alpha as Ridge/Lasso penalty (ignored for OLS)",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--sweep-models",
        type=parse_models,
        default=None,
        help="comma-separated models on validation grid, example ols,ridge,lasso",
    )
    p.add_argument(
        "--ridge-alphas",
        type=parse_float_list,
        default=parse_float_list("0.1,1.0,10.0"),
        help="comma-separated Ridge alphas for ridge rows in sweep",
    )
    p.add_argument(
        "--lasso-alphas",
        type=parse_float_list,
        default=parse_float_list("0.0001,0.0003,0.001,0.003,0.01"),
        help="comma-separated Lasso alphas for lasso rows in sweep",
    )
    p.add_argument(
        "--tune-split",
        type=float,
        default=0.2,
        help="validation fraction held out from train only for sweep selection; 0 disables validation sweep",
    )
    p.add_argument(
        "--refit-best-on-train-plus-val",
        action="store_true",
        help="after validation selection, refit final model on all train rows before test",
    )
    p.add_argument("--sweep-results-out", default=None,
                   help="optional json path for sweep results")
    p.add_argument(
        "--per-column-metrics-csv",
        type=Path,
        default=None,
        help="CSV: одна строка на таргет-колонку music_data (mae, rmse, r2, pearson, …) на тесте",
    )
    p.add_argument(
        "--skip-experiment-log",
        action="store_true",
        help="не писать строку в text_music_train_experiments",
    )
    p.add_argument("--top-percent", type=float, default=35.0)
    register_retrieval_column_mask_cli(p)
    p.add_argument(
        "--music-mode",
        choices=("first", "second", "avg_loss", "min_loss"),
        default="min_loss",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    validate_pg_identifier(args.rubert_table)
    validate_pg_identifier(args.embedding_column)
    validate_pg_identifier(args.music_table)

    if not 0.0 < args.test_fraction < 1.0:
        print("--test-fraction must be between 0 and 1")
        return 2
    if args.tune_split != 0.0 and not 0.0 < args.tune_split < 1.0:
        print("--tune-split must be 0 or between 0 and 1")
        return 2
    if args.top_percent <= 0:
        print("--top-percent must be > 0")
        return 2

    music_mode_note(args.music_mode)
    dsn = args.dsn or default_dsn()

    with psycopg.connect(dsn) as conn:
        loaded = load_frames(conn, args)
        if loaded is None:
            return 2

        err_rb = validate_held_out_game_row_bounds(args)
        if err_rb is not None:
            return err_rb

        held_out_games: list[str] | None = None
        if args.split == "within_game":
            train_idx, test_idx = split_within_game(
                loaded.rows, args.test_fraction, args.seed)
        else:
            if args.held_out_games_seed is None:
                args.held_out_games_seed = secrets.randbelow(1 << 31)
                print(
                    "held_out_games_seed random",
                    int(args.held_out_games_seed),
                    "repeat with --held-out-games-seed",
                    int(args.held_out_games_seed),
                )
            else:
                print("held_out_games_seed fixed",
                      int(args.held_out_games_seed))
            trio = split_leave_games_out_from_args(loaded.rows, args)
            if trio is None:
                return 2
            train_idx, test_idx, held_out_games = trio

        if not train_idx or not test_idx:
            print("empty train or test split", "train",
                  len(train_idx), "test", len(test_idx))
            return 2
        if set(train_idx) & set(test_idx):
            print("train and test overlap; aborting")
            return 2

        print_split_visibility(loaded.rows, train_idx,
                               test_idx, args, held_out_games)

        if args.sweep_models is None:
            combos = [
                SweepCombo(
                    model=args.model,
                    alpha=None
                    if args.model == "ols"
                    else args.alpha,
                ),
            ]
        else:
            combos = sweep_combos(
                args.sweep_models,
                args.ridge_alphas,
                args.lasso_alphas,
            )
        if not combos:
            print("no sweep combos")
            return 2
        n_ols = sum(1 for c in combos if c.model == "ols")
        n_ridge = sum(1 for c in combos if c.model == "ridge")
        n_lasso = sum(1 for c in combos if c.model == "lasso")
        print(
            "sweep grid size",
            len(combos),
            "breakdown ols",
            n_ols,
            "ridge",
            n_ridge,
            "lasso",
            n_lasso,
            "note: one OLS combo, one row per ridge alpha, one row per lasso alpha",
        )
        effective_tune_split = args.tune_split if len(combos) > 1 else 0.0
        tune_train_idx, val_idx = split_validation_from_train(
            train_idx,
            effective_tune_split,
            args.seed + 1,
        )
        if args.tune_split and len(combos) == 1:
            print("validation split skipped; only one config")
        if effective_tune_split and not val_idx:
            print("validation split empty; using train split without tuning holdout")
        print("validation rows", len(val_idx), "games covered",
              game_count(loaded.rows, val_idx))
        print("fit rows for tuning", len(tune_train_idx),
              "games covered", game_count(loaded.rows, tune_train_idx))

        sweep_results: list[dict[str, object]] = []
        best_combo = combos[0]
        best_score = float("inf")
        for combo in combos:
            alpha_for_fit = combo.alpha if combo.alpha is not None else args.alpha
            print(
                "tuning combo",
                "model",
                combo.model,
                "alpha",
                combo.alpha if combo.alpha is not None else "none",
            )
            t_fit0 = time.perf_counter()
            model = build_model(combo.model, alpha_for_fit, rng_seed=args.seed)
            model.fit(loaded.x[tune_train_idx], loaded.y[tune_train_idx])
            print("tune_fit_wall_seconds", round(
                time.perf_counter() - t_fit0, 4))
            if val_idx:
                y_val_pred = np.asarray(model.predict(
                    loaded.x[val_idx]), dtype=np.float32)
                val_metrics = print_regression_block(
                    "Tuning metrics ON VALIDATION ONLY",
                    loaded.y_columns,
                    loaded.y[val_idx],
                    y_val_pred,
                    False,
                )
                score = float(val_metrics["mae_macro"])
            else:
                val_metrics = {}
                score = 0.0
            result = {
                "model": combo.model,
                "alpha": combo.alpha,
                "val_mae_macro": score if val_idx else None,
                "val_metrics": val_metrics,
            }
            sweep_results.append(result)
            print(
                "sweep combo",
                "model",
                combo.model,
                "alpha",
                combo.alpha if combo.alpha is not None else "none",
                "val_mae_macro",
                round(score, 6) if val_idx else "none",
            )
            if score < best_score:
                best_score = score
                best_combo = combo

        final_train_idx = train_idx if args.refit_best_on_train_plus_val or not val_idx else tune_train_idx
        final_alpha = best_combo.alpha if best_combo.alpha is not None else args.alpha
        t_final0 = time.perf_counter()
        final_model = build_model(
            best_combo.model, final_alpha, rng_seed=args.seed)
        final_model.fit(loaded.x[final_train_idx], loaded.y[final_train_idx])
        final_fit_wall_s = time.perf_counter() - t_final0
        print("final_fit_wall_seconds", round(final_fit_wall_s, 4))
        y_pred = np.asarray(final_model.predict(
            loaded.x[test_idx]), dtype=np.float32)

        y_train_fit_pred = np.asarray(
            final_model.predict(loaded.x[final_train_idx]),
            dtype=np.float32,
        )
        y_val_fit_pred = None
        if val_idx:
            y_val_fit_pred = np.asarray(
                final_model.predict(loaded.x[val_idx]),
                dtype=np.float32,
            )
        train_metrics_summary = print_regression_block(
            "Regression metrics ON TRAIN (fit set)",
            loaded.y_columns,
            loaded.y[final_train_idx],
            y_train_fit_pred,
            False,
        )
        val_metrics_summary = None
        if val_idx and y_val_fit_pred is not None:
            val_metrics_summary = print_regression_block(
                "Regression metrics ON VALIDATION (final model)",
                loaded.y_columns,
                loaded.y[val_idx],
                y_val_fit_pred,
                False,
            )

        print("")
        print(
            "best config",
            "model",
            best_combo.model,
            "alpha",
            best_combo.alpha if best_combo.alpha is not None else "none",
            "selected_by",
            "validation_mae_macro" if val_idx else "single_config_no_validation",
            "score",
            round(best_score, 6) if val_idx else "none",
        )
        print("final fit rows", len(final_train_idx), "test rows", len(test_idx))
        test_metrics = print_regression_block(
            "Regression metrics ON TEST ONLY",
            loaded.y_columns,
            loaded.y[test_idx],
            y_pred,
            False,
        )

        if args.per_column_metrics_csv is not None:
            write_per_column_metrics_csv(
                args.per_column_metrics_csv,
                y_columns=loaded.y_columns,
                y_true=loaded.y[test_idx],
                y_pred=y_pred,
                meta={
                    "embedding_column": args.embedding_column,
                    "rubert_table": args.rubert_table,
                    "music_table": args.music_table,
                    "y_group": args.y_group,
                    "split": args.split,
                    "seed": str(args.seed),
                    "test_fraction": str(args.test_fraction),
                    "test_games": str(args.test_games),
                    "test_rows": str(len(test_idx)),
                    "merge_identical_embeddings": str(
                        bool(args.merge_identical_embeddings),
                    ),
                    "final_model": str(best_combo.model),
                    "final_alpha": ""
                    if best_combo.alpha is None
                    else str(best_combo.alpha),
                },
            )
            print("wrote per-column metrics csv",
                  str(args.per_column_metrics_csv))

        print("")
        print("retrieval hit rates")
        from text_scripts.ml_scripts.retrieval_column_mask import (
            prepare_retrieval_column_indices,
        )
        r_col_idx, _r_mask_detail = prepare_retrieval_column_indices(
            args,
            y_true_val=loaded.y[val_idx] if val_idx else None,
            y_pred_val=y_val_fit_pred if val_idx else None,
            y_true_test=loaded.y[test_idx],
            y_pred_test=y_pred,
        )
        retrieval = retrieval_hit_rates(
            conn,
            args.music_table,
            loaded.y_columns,
            [loaded.rows[i] for i in test_idx],
            y_pred,
            args.top_percent,
            r_col_idx,
        )
        for metric, (hit, total, skipped) in retrieval.items():
            rate = float(hit) / float(total) if total else float("nan")
            print(
                metric,
                "hit@",
                args.top_percent,
                "percent",
                round(rate, 6) if total else "nan",
                "hits",
                hit,
                "total",
                total,
                "skipped_single_or_missing",
                skipped,
            )

        if not args.skip_experiment_log:
            from text_scripts.ml_scripts.db_consts import TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE
            from text_scripts.ml_scripts.train_experiment_log import log_linear_run

            sm = getattr(args, "sweep_models", None)
            sweep_str = ",".join(sm) if sm else None
            log_linear_run(
                conn,
                argv_text=" ".join(sys.argv),
                args=args,
                n_rows_loaded=len(loaded.rows),
                n_y_columns=len(loaded.y_columns),
                n_train_fit=len(final_train_idx),
                n_val=len(val_idx),
                n_test=len(test_idx),
                train_metrics=train_metrics_summary,
                val_metrics=val_metrics_summary,
                test_metrics=test_metrics,
                retrieval=retrieval,
                final_fit_wall_s=float(final_fit_wall_s),
                sweep_models_str=sweep_str,
                best_model=str(best_combo.model),
                best_alpha=float(
                    best_combo.alpha) if best_combo.alpha is not None else None,
            )
            print("logged experiment to", TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE)

        if args.sweep_results_out:
            payload = {
                "seed": args.seed,
                "split": args.split,
                "test_fraction": args.test_fraction,
                "test_games": args.test_games,
                "tune_split": args.tune_split,
                "refit_best_on_train_plus_val": bool(args.refit_best_on_train_plus_val),
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "val_rows": len(val_idx),
                "best_config": {
                    "model": best_combo.model,
                    "alpha": best_combo.alpha,
                },
                "test_metrics": test_metrics,
                "sweep_results": sweep_results,
            }
            out_path = Path(args.sweep_results_out)
            out_path.write_text(json.dumps(
                payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("wrote sweep results", str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
