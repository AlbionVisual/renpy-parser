from __future__ import annotations

import json
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import psycopg
from psycopg.types.json import Json

from text_scripts.ml_scripts.db_consts import (
    CREATE_TEXT_MUSIC_TRAIN_EXPERIMENTS_SQL,
    TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE,
)


def _args_to_jsonable(args: Namespace) -> dict[str, Any]:
    d = vars(args).copy()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
        if k == "hidden" and v is not None and not isinstance(v, (str, bytes, int, float, bool)):
            try:
                d[k] = list(v)
            except TypeError:
                d[k] = str(v)
    return d


def _split_metrics(prefix: str, s: Mapping[str, float] | None) -> dict[str, Any]:
    keys = (
        "mae_macro",
        "rmse_macro",
        "r2_macro",
        "evs_macro",
        "medae_macro",
        "row_cosine_mean",
        "column_pearson_mean",
        "max_abs_macro",
    )
    if s is None:
        return {f"{prefix}_{k}": None for k in keys}
    return {f"{prefix}_{k}": float(s[k]) for k in keys}


def retrieval_rates_row(
    retrieval: dict[str, tuple[int, int, int]],
    top_percent: float,
) -> dict[str, Any]:
    rmse_h, rmse_t, sk = retrieval["rmse"]
    mae_h, mae_t, _ = retrieval["mae"]
    cos_h, cos_t, _ = retrieval["cosine"]
    return {
        "retrieval_top_percent": float(top_percent),
        "retrieval_rmse_hit_rate": float(rmse_h) / float(rmse_t) if rmse_t else None,
        "retrieval_mae_hit_rate": float(mae_h) / float(mae_t) if mae_t else None,
        "retrieval_cosine_hit_rate": float(cos_h) / float(cos_t) if cos_t else None,
        "retrieval_rmse_hits": int(rmse_h),
        "retrieval_rmse_total": int(rmse_t),
        "retrieval_mae_hits": int(mae_h),
        "retrieval_mae_total": int(mae_t),
        "retrieval_cosine_hits": int(cos_h),
        "retrieval_cosine_total": int(cos_t),
        "retrieval_skipped": int(sk),
    }


def ensure_train_experiments_table(cur) -> None:
    cur.execute(CREATE_TEXT_MUSIC_TRAIN_EXPERIMENTS_SQL)


def insert_train_experiment_row(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    cols = sorted(row.keys())
    vals = [row[c] for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    q = "insert into {} ({}) values ({})".format(
        TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE,
        ", ".join(cols),
        placeholders,
    )
    try:
        with conn.cursor() as cur:
            ensure_train_experiments_table(cur)
            cur.execute(q, vals)
        conn.commit()
    except Exception as exc:
        print(
            "insert_train_experiment_row failed",
            type(exc).__name__,
            str(exc),
        )
        traceback.print_exc()


def log_linear_run(
    conn: psycopg.Connection,
    *,
    argv_text: str,
    args: Namespace,
    n_rows_loaded: int,
    n_y_columns: int,
    n_train_fit: int,
    n_val: int,
    n_test: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float] | None,
    test_metrics: Mapping[str, float],
    retrieval: dict[str, tuple[int, int, int]],
    final_fit_wall_s: float,
    sweep_models_str: str | None,
    best_model: str,
    best_alpha: float | None,
) -> None:
    row: dict[str, Any] = {
        "script": "linear",
        "argv_text": argv_text,
        "config_json": Json(_args_to_jsonable(args)),
        "embedding_column": args.embedding_column,
        "rubert_table": args.rubert_table,
        "music_table": args.music_table,
        "y_group": args.y_group,
        "split_mode": args.split,
        "merge_identical_embeddings": bool(args.merge_identical_embeddings),
        "test_fraction": float(args.test_fraction),
        "test_games": int(args.test_games),
        "seed": int(args.seed),
        "n_rows_loaded": int(n_rows_loaded),
        "n_y_columns": int(n_y_columns),
        "n_train_fit": int(n_train_fit),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "val_tune_fraction": float(args.tune_split),
        "top_percent": float(args.top_percent),
        "linear_model": best_model,
        "linear_alpha": float(best_alpha) if best_alpha is not None else None,
        "linear_sweep_models": sweep_models_str,
        "linear_refit_best_on_train_plus_val": bool(args.refit_best_on_train_plus_val),
        "linear_final_fit_wall_s": float(final_fit_wall_s),
        "mlp_hidden": None,
        "mlp_dropout": None,
        "mlp_lr": None,
        "mlp_weight_decay": None,
        "mlp_epochs_ran": None,
        "mlp_batch_size": None,
        "mlp_patience": None,
        "mlp_standardize_x": None,
        "mlp_standardize_y": None,
        "mlp_best_val_mae_norm_macro": None,
        "mlp_train_wall_s": None,
    }
    row.update(_split_metrics("train", train_metrics))
    row.update(_split_metrics("val", val_metrics))
    row.update(_split_metrics("test", test_metrics))
    row.update(retrieval_rates_row(retrieval, args.top_percent))
    insert_train_experiment_row(conn, row)


def log_mlp_run(
    conn: psycopg.Connection,
    *,
    argv_text: str,
    args: Namespace,
    n_rows_loaded: int,
    n_y_columns: int,
    n_train_fit: int,
    n_val: int,
    n_test: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float] | None,
    test_metrics: Mapping[str, float],
    retrieval: dict[str, tuple[int, int, int]],
    train_wall_s: float,
    epochs_ran: int,
    best_val_mae_norm: float,
    script: str = "mlp",
) -> None:
    row: dict[str, Any] = {
        "script": script,
        "argv_text": argv_text,
        "config_json": Json(_args_to_jsonable(args)),
        "embedding_column": args.embedding_column,
        "rubert_table": args.rubert_table,
        "music_table": args.music_table,
        "y_group": args.y_group,
        "split_mode": args.split,
        "merge_identical_embeddings": bool(args.merge_identical_embeddings),
        "test_fraction": float(args.test_fraction),
        "test_games": int(args.test_games),
        "seed": int(args.seed),
        "n_rows_loaded": int(n_rows_loaded),
        "n_y_columns": int(n_y_columns),
        "n_train_fit": int(n_train_fit),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "val_tune_fraction": float(args.val_fraction),
        "top_percent": float(args.top_percent),
        "linear_model": None,
        "linear_alpha": None,
        "linear_sweep_models": None,
        "linear_refit_best_on_train_plus_val": None,
        "linear_final_fit_wall_s": None,
        "mlp_hidden": ",".join(str(h) for h in args.hidden),
        "mlp_dropout": float(args.dropout),
        "mlp_lr": float(args.lr),
        "mlp_weight_decay": float(args.weight_decay),
        "mlp_epochs_ran": int(epochs_ran),
        "mlp_batch_size": int(args.batch_size),
        "mlp_patience": int(args.patience),
        "mlp_standardize_x": bool(args.standardize_x),
        "mlp_standardize_y": bool(not args.no_standardize_y),
        "mlp_best_val_mae_norm_macro": float(best_val_mae_norm)
        if best_val_mae_norm < float("inf")
        else None,
        "mlp_train_wall_s": float(train_wall_s),
    }
    row.update(_split_metrics("train", train_metrics))
    row.update(_split_metrics("val", val_metrics))
    row.update(_split_metrics("test", test_metrics))
    row.update(retrieval_rates_row(retrieval, args.top_percent))
    insert_train_experiment_row(conn, row)
