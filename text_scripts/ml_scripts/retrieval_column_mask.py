from __future__ import annotations

import argparse
import re
from argparse import ArgumentParser
from typing import Any

import numpy as np


def _column_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from text_scripts.ml_scripts.train_rubert_to_music_linear import (
        column_pearson as cp,
    )

    return cp(y_true, y_pred)


def register_retrieval_column_mask_cli(p: ArgumentParser) -> None:
    p.add_argument(
        "--retrieval-column-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="если включено, расстояния retrieval (rmse/mae/cosine) считаются только по столбцам, "
        "отфильтрованным по --retrieval-column-filter на выбранном сплите",
    )
    p.add_argument(
        "--retrieval-column-filter",
        type=str,
        default="",
        help="пример: pearson>=0.2||r2>0.05 — ИЛИ между группами; внутри группы: mae<0.8&&rmse<1 — И. "
        "скобки не поддерживаются; сложное (A||B)&& C задавайте как (A&&C)||(B&&C). "
        "метрики: pearson, mae, rmse, mse, r2 (по столбцам на сплите маски). "
        "pearson: 2..100 -> /100. добор столбцов: см. --retrieval-mask-min-columns",
    )
    p.add_argument(
        "--retrieval-mask-from-split",
        choices=("val", "test"),
        default="val",
        help="на каком сплите считать per-column статистики для фильтра столбцов",
    )
    p.add_argument(
        "--retrieval-mask-min-columns",
        type=int,
        default=10,
        help="минимум столбцов в маске; если фильтр даёт меньше — добор по --retrieval-mask-fallback-percentile "
        "и при необходимости по рангу pearson",
    )
    p.add_argument(
        "--retrieval-mask-fallback-percentile",
        type=float,
        default=80.0,
        help="перцентиль по pearson (по столбцам на выбранном сплите): сначала берутся столбцы с "
        "pearson >= этому значению, пока не наберётся min_columns; затем при нехватке — лучшие по pearson",
    )


def _per_column_mae_rmse_mse(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = (y_pred.astype(np.float64, copy=False)
            - y_true.astype(np.float64, copy=False))
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff * diff, axis=0)
    rmse = np.sqrt(mse)
    return mae, rmse, mse


def _column_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    ss_res = np.sum(diff * diff, axis=0)
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=0)
    r2 = np.full(int(y_true.shape[1]), np.nan, dtype=np.float64)
    m = ss_tot > 0
    r2[m] = 1.0 - ss_res[m] / ss_tot[m]
    return r2


def per_column_stats_arrays(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    mae, rmse, mse = _per_column_mae_rmse_mse(y_true, y_pred)
    pearson = _column_pearson(y_true, y_pred)
    r2 = _column_r2(y_true, y_pred)
    return {"pearson": pearson, "mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


_TOKEN_RE = re.compile(
    r"^\s*(pearson|mae|rmse|mse|r2)\s*(<=|>=|==|!=|<|>)\s*([-+eE0-9.]+)\s*$",
    re.IGNORECASE,
)


def _norm_pearson_threshold(v: float) -> float:
    if 2.0 < v <= 100.0:
        return v / 100.0
    return v


def _comparison_mask(
    a: np.ndarray,
    op: str,
    thr: float,
    metric_key: str,
) -> np.ndarray:
    if metric_key == "pearson":
        thr = _norm_pearson_threshold(thr)
    vc = np.asarray(a, dtype=np.float64)
    if op == ">":
        ok = vc > thr
    elif op == ">=":
        ok = vc >= thr
    elif op == "<":
        ok = vc < thr
    elif op == "<=":
        ok = vc <= thr
    elif op == "==":
        ok = vc == thr
    else:
        ok = vc != thr
    return ok & np.isfinite(vc)


def _column_mask_and_group(stats: dict[str, np.ndarray], and_expr: str) -> np.ndarray:
    parts = [p.strip() for p in and_expr.replace("&&", "&").split("&") if p.strip()]
    if not parts:
        raise ValueError("empty && group in filter: " + repr(and_expr))
    n = None
    for k in stats:
        n = int(stats[k].shape[0])
        break
    if n is None:
        raise ValueError("no stat arrays")
    mask = np.ones(n, dtype=bool)
    for part in parts:
        m = _TOKEN_RE.match(part)
        if not m:
            raise ValueError("bad condition (e.g. pearson>=0.3): " + repr(part))
        key = m.group(1).lower()
        op = m.group(2)
        thr = float(m.group(3))
        if key not in stats:
            raise ValueError("unknown metric " + repr(key))
        arr = stats[key]
        if arr.shape[0] != n:
            raise ValueError("stat length mismatch")
        mask &= _comparison_mask(arr, op, thr, key)
    return mask


def column_mask_from_filter_string(stats: dict[str, np.ndarray], expr: str) -> np.ndarray:
    s = expr.strip()
    if not s:
        raise ValueError("empty retrieval column filter")
    n = None
    for k in stats:
        n = int(stats[k].shape[0])
        break
    if n is None:
        raise ValueError("no stat arrays")
    or_parts = [p.strip() for p in re.split(r"\|\|", s) if p.strip()]
    if not or_parts:
        raise ValueError("no conditions in filter")
    out = np.zeros(n, dtype=bool)
    for grp in or_parts:
        out |= _column_mask_and_group(stats, grp)
    return out


def _print_stat_quantiles(stats: dict[str, np.ndarray], split: str) -> None:
    print("retrieval filter diag: per-column quantiles on split", split)
    for name in ("pearson", "mae", "rmse", "mse", "r2"):
        a = np.asarray(stats[name], dtype=np.float64)
        aa = a[np.isfinite(a)]
        if aa.size == 0:
            print(" ", name, "all_nan_or_nonfinite")
            continue
        print(
            " ",
            name,
            "p25",
            round(float(np.percentile(aa, 25)), 6),
            "p50",
            round(float(np.percentile(aa, 50)), 6),
            "p75",
            round(float(np.percentile(aa, 75)), 6),
            "p80",
            round(float(np.percentile(aa, 80)), 6),
            "max",
            round(float(np.max(aa)), 6),
        )


def _finalize_column_indices(
    stats: dict[str, np.ndarray],
    filt: str,
    *,
    split: str,
    min_k: int,
    pct: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    p = np.asarray(stats["pearson"], dtype=np.float64)
    n_dim = int(p.size)
    min_k = max(1, min(int(min_k), n_dim))
    finite_p = np.isfinite(p)
    ranked = np.argsort(-np.nan_to_num(p, nan=-np.inf))

    chosen: set[int] = set()
    n_from_filter = 0
    filter_parse_error: str | None = None

    if filt.strip():
        try:
            mask = column_mask_from_filter_string(stats, filt)
            chosen = set(np.flatnonzero(mask).tolist())
            n_from_filter = len(chosen)
        except ValueError as exc:
            filter_parse_error = str(exc)
            print(
                "retrieval column mask WARNING: filter failed",
                repr(filt),
                str(exc),
            )
            _print_stat_quantiles(stats, split)
    else:
        print(
            "retrieval column mask WARNING: empty --retrieval-column-filter; "
            "using only pearson percentile / rank fallback",
        )

    p_thr: float | None = None
    if finite_p.any():
        p_thr = float(np.percentile(p[finite_p], float(pct)))
    else:
        p_thr = None

    n_from_pct = 0
    if len(chosen) < min_k and p_thr is not None:
        before = len(chosen)
        for j in ranked:
            if len(chosen) >= min_k:
                break
            jj = int(j)
            if jj in chosen:
                continue
            if finite_p[jj] and float(p[jj]) >= p_thr:
                chosen.add(jj)
        n_from_pct = len(chosen) - before

    n_from_rank = 0
    if len(chosen) < min_k:
        before = len(chosen)
        for j in ranked:
            if len(chosen) >= min_k:
                break
            chosen.add(int(j))
        n_from_rank = len(chosen) - before

    used_all = False
    if len(chosen) < min_k:
        chosen = set(range(n_dim))
        used_all = True
        print(
            "retrieval column mask WARNING: cannot reach min_k",
            min_k,
            "with dim",
            n_dim,
            "— using all columns",
        )

    idx = np.array(sorted(chosen), dtype=np.int64)

    print("")
    print("retrieval column mask SUMMARY")
    print("  split_for_stats:", split)
    print("  filter:", repr(filt) if filt else "(none)")
    if filter_parse_error:
        print("  filter_error:", filter_parse_error)
    print("  columns_matched_filter:", n_from_filter)
    print(
        "  columns_added_pearson_ge_p" + str(int(pct)) + ":",
        n_from_pct,
        "pearson_threshold",
        None if p_thr is None else round(float(p_thr), 6),
    )
    print("  columns_added_pearson_rank_topup:", n_from_rank)
    print("  final_column_count:", int(idx.size), "of", n_dim)
    print("  used_all_columns:", used_all)

    meta: dict[str, Any] = {
        "retrieval_mask_split": split,
        "retrieval_filter_string": filt or None,
        "retrieval_filter_parse_error": filter_parse_error,
        "retrieval_n_columns_total": n_dim,
        "retrieval_mask_min_columns_effective": min_k,
        "retrieval_mask_fallback_percentile": float(pct),
        "retrieval_n_from_filter": int(n_from_filter),
        "retrieval_n_from_percentile_bucket": int(n_from_pct),
        "retrieval_n_from_rank_topup": int(n_from_rank),
        "retrieval_pearson_percentile_threshold": p_thr,
        "retrieval_pearson_percentile_level": float(pct),
        "retrieval_n_columns_selected": int(idx.size),
        "retrieval_mask_used_all_columns": bool(used_all),
    }
    return idx, meta


def prepare_retrieval_column_indices(
    args: Any,
    *,
    y_true_val: np.ndarray | None,
    y_pred_val: np.ndarray | None,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    out_meta: dict[str, Any] = {}
    if not bool(getattr(args, "retrieval_column_mask", False)):
        return None, out_meta

    filt = str(getattr(args, "retrieval_column_filter", "") or "").strip()
    min_k = int(getattr(args, "retrieval_mask_min_columns", 10))
    pct = float(getattr(args, "retrieval_mask_fallback_percentile", 80.0))
    if pct <= 0.0 or pct > 100.0:
        pct = 80.0
        print(
            "retrieval column mask WARNING: fallback percentile out of (0,100], using 80",
        )

    split = str(getattr(args, "retrieval_mask_from_split", "val"))
    if split == "val":
        if (
            y_true_val is None
            or y_pred_val is None
            or len(y_true_val) == 0
        ):
            print(
                "retrieval column mask WARNING: no val set; using test for filter stats",
            )
            split = "test"
            y_t, y_p = y_true_test, y_pred_test
        else:
            y_t, y_p = y_true_val, y_pred_val
    else:
        y_t, y_p = y_true_test, y_pred_test

    stats = per_column_stats_arrays(y_t, y_p)

    idx, detail = _finalize_column_indices(
        stats,
        filt,
        split=split,
        min_k=min_k,
        pct=pct,
    )
    out_meta.update(detail)
    return idx, out_meta


def retrieval_mask_meta(
    args: Any, col_idx: np.ndarray | None, detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "retrieval_column_mask": bool(getattr(args, "retrieval_column_mask", False)),
        "retrieval_column_filter": (getattr(args, "retrieval_column_filter", "") or None),
        "retrieval_mask_from_split": getattr(args, "retrieval_mask_from_split", None),
        "retrieval_n_columns": int(col_idx.size) if col_idx is not None else None,
        "retrieval_mask_min_columns": int(getattr(args, "retrieval_mask_min_columns", 10)),
        "retrieval_mask_fallback_percentile": float(
            getattr(args, "retrieval_mask_fallback_percentile", 80.0)),
    }
    if detail:
        for k, v in detail.items():
            if k not in base:
                base[k] = v
    return base
