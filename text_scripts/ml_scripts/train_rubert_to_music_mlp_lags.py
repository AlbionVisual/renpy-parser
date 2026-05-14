from __future__ import annotations
import argparse
import math
import secrets
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_ml = Path(__file__).resolve().parent
_repo_root = _ml.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from text_scripts.ml_scripts.dataset_files import (
    apply_merge_identical_embeddings_if_requested,
    load_candidates_dir,
    load_dataset_dir,
    retrieval_hit_rates_from_candidates,
    training_db_context,
)
from text_scripts.ml_scripts.rubert_embeddings import validate_pg_identifier
from text_scripts.ml_scripts.run_artifacts import (
    RunArtifacts,
    apply_run_dir_defaults,
    write_metrics_json,
    write_run_config_json,
)
from text_scripts.ml_scripts.train_rubert_to_music_linear import (
    PairRow,
    game_count,
    load_frames,
    music_mode_note,
    print_regression_block,
    print_split_visibility,
    retrieval_hit_rates,
    split_leave_games_out_from_args,
    validate_held_out_game_row_bounds,
    split_validation_from_train,
    split_validation_games_held_out_from_train,
    split_validation_tail_per_game_from_train,
    split_within_game,
    write_per_column_metrics_csv,
)
from text_scripts.ml_scripts.mlp_train_defaults import register_shared_rubert_mlp_cli
from text_scripts.ml_scripts.train_rubert_to_music_mlp import (
    MLP,
    denormalize_y,
    parse_hidden,
    pick_device,
    standardize,
    train_mlp,
)


def build_concat_embedding_lags(
    rows: Sequence[PairRow],
    x: np.ndarray,
    n_prev_embeddings: int,
) -> np.ndarray:
    if n_prev_embeddings < 0:
        raise ValueError("n_prev_embeddings must be >= 0")
    n, d = int(x.shape[0]), int(x.shape[1])
    blocks = 1 + n_prev_embeddings
    out = np.zeros((n, d * blocks), dtype=np.float32)
    z = np.zeros(d, dtype=np.float32)
    for i in range(n):
        g = rows[i].game
        parts: list[np.ndarray] = [x[i]]
        for ell in range(1, n_prev_embeddings + 1):
            j = i - ell
            if j >= 0 and rows[j].game == g:
                parts.append(x[j])
            else:
                parts.append(z)
        out[i] = np.concatenate(parts, axis=0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLP: RuBERT + concatenated same-game embedding lags -> music_data.",
    )
    register_shared_rubert_mlp_cli(
        p,
        parse_hidden=parse_hidden,
        device_default="cpu",
        include_embedding_lags=True,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_run_dir_defaults(args)
    if args.dataset_dir is not None:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
        args.skip_experiment_log = True
        print("dataset-dir set: skip_experiment_log forced True")
    if args.dataset_dir is None:
        validate_pg_identifier(args.rubert_table)
        validate_pg_identifier(args.embedding_column)
        validate_pg_identifier(args.music_table)
    if args.embedding_lags < 0:
        print("--embedding-lags must be >= 0")
        return 2
    if not 0.0 < args.test_fraction < 1.0:
        print("--test-fraction must be between 0 and 1")
        return 2
    if args.val_fraction != 0.0 and not 0.0 < args.val_fraction < 1.0:
        print("--val-fraction must be 0 or between 0 and 1")
        return 2
    if args.top_percent <= 0:
        print("--top-percent must be > 0")
        return 2
    if args.epochs < 1:
        print("--epochs must be >= 1")
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1")
        return 2
    if args.overfit_gap_threshold > 0.0 and args.overfit_patience < 1:
        print("--overfit-patience must be >= 1 when --overfit-gap-threshold > 0")
        return 2
    if args.val_split == "game" and args.val_games < 1:
        print("--val-games must be >= 1 when --val-split game")
        return 2

    err_rb = validate_held_out_game_row_bounds(args)
    if err_rb is not None:
        return err_rb

    music_mode_note(args.music_mode)
    show_progress = not args.no_progress
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    with RunArtifacts(getattr(args, "run_dir", None)):
        with training_db_context(args) as conn:
            if args.dataset_dir is not None:
                loaded = load_dataset_dir(args.dataset_dir)
                loaded = apply_merge_identical_embeddings_if_requested(
                    loaded, bool(args.merge_identical_embeddings)
                )
            else:
                loaded = load_frames(conn, args)
            if loaded is None:
                return 2
            if args.run_dir is not None:
                write_run_config_json(
                    args.run_dir,
                    args,
                    extra={
                        "n_rows": len(loaded.rows),
                        "x_dim": int(loaded.x.shape[1]),
                        "y_dim": int(loaded.y.shape[1]),
                    },
                )

            x_lag = build_concat_embedding_lags(
                loaded.rows,
                loaded.x,
                int(args.embedding_lags),
            )
            print(
                "mlp_lags raw_emb_dim",
                loaded.x.shape[1],
                "embedding_lags",
                int(args.embedding_lags),
                "concat_in_dim",
                x_lag.shape[1],
                "rows",
                len(loaded.rows),
            )

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

            args.tune_split = args.val_fraction
            args.refit_best_on_train_plus_val = False
            print_split_visibility(loaded.rows, train_idx,
                                   test_idx, args, held_out_games)
            if args.val_split == "game":
                print("mlp_lags val_split game: --val-games задаёт число игр в val")
                tune_train_idx, val_idx = split_validation_games_held_out_from_train(
                    loaded.rows,
                    train_idx,
                    args.val_games,
                    args.seed + 1,
                )
                if not val_idx:
                    print(
                        "warning: val-split game failed; no val",
                    )
            elif args.val_split == "tail":
                tune_train_idx, val_idx = split_validation_tail_per_game_from_train(
                    loaded.rows,
                    train_idx,
                    args.val_fraction,
                )
                if args.val_fraction and not val_idx:
                    print("warning: val-split tail empty")
            else:
                tune_train_idx, val_idx = split_validation_from_train(
                    train_idx,
                    args.val_fraction,
                    args.seed + 1,
                )
                if args.val_fraction and not val_idx:
                    print("validation split empty")

            print(
                "mlp_lags val_split",
                args.val_split,
                "tune rows",
                len(tune_train_idx),
                "val rows",
                len(val_idx),
                "games tune",
                game_count(loaded.rows, tune_train_idx),
                "games val",
                game_count(loaded.rows, val_idx),
            )
            st = set(tune_train_idx)
            sv = set(val_idx)
            ste = set(test_idx)
            print(
                "mlp_lags index_overlap tune_val",
                len(st & sv),
                "tune_test",
                len(st & ste),
                "val_test",
                len(sv & ste),
            )
            if st & sv or st & ste or sv & ste:
                print("fatal: index overlap")
                return 2

            x_tr = x_lag[tune_train_idx]
            y_tr = loaded.y[tune_train_idx]
            if args.standardize_x:
                x_tr_n, x_mean, x_std = standardize(x_tr)
                x_fit_mean, x_fit_std = x_mean, x_std
            else:
                x_tr_n = x_tr.astype(np.float32)
                x_fit_mean = x_fit_std = None

            if args.no_standardize_y:
                y_tr_n = y_tr.astype(np.float32)
                y_mean = np.zeros(y_tr.shape[1], dtype=np.float32)
                y_std = np.ones(y_tr.shape[1], dtype=np.float32)
            else:
                y_tr_n, y_mean, y_std = standardize(y_tr)

            if val_idx:
                xv = x_lag[val_idx]
                if args.standardize_x and x_fit_mean is not None:
                    xv = ((xv.astype(np.float64) - x_fit_mean) /
                          x_fit_std).astype(np.float32)
                else:
                    xv = xv.astype(np.float32)
                yv = loaded.y[val_idx]
                if args.no_standardize_y:
                    yv_n = yv.astype(np.float32)
                else:
                    yv_n = ((yv.astype(np.float64) - y_mean) /
                            y_std).astype(np.float32)
                use_es = True
            else:
                xv = np.zeros((0, x_tr_n.shape[1]), dtype=np.float32)
                yv_n = np.zeros((0, y_tr_n.shape[1]), dtype=np.float32)
                use_es = False

            x_test_np = x_lag[test_idx]
            if args.standardize_x and x_fit_mean is not None:
                x_test_n = ((x_test_np.astype(np.float64) -
                            x_fit_mean) / x_fit_std).astype(np.float32)
            else:
                x_test_n = x_test_np.astype(np.float32)
            y_te = loaded.y[test_idx]
            if args.no_standardize_y:
                y_te_n = y_te.astype(np.float32)
            else:
                y_te_n = ((y_te.astype(np.float64) - y_mean) /
                          y_std).astype(np.float32)

            x_train_t = torch.from_numpy(x_tr_n)
            y_train_t = torch.from_numpy(y_tr_n)
            x_val_t = torch.from_numpy(xv)
            y_val_t = torch.from_numpy(yv_n)
            if args.log_test_mae_norm_each_epoch:
                x_test_mon_t = torch.from_numpy(x_test_n)
                y_test_mon_t = torch.from_numpy(y_te_n)
            else:
                x_test_mon_t = torch.zeros(
                    (0, x_tr_n.shape[1]), dtype=torch.float32)
                y_test_mon_t = torch.zeros(
                    (0, y_tr_n.shape[1]), dtype=torch.float32)

            in_dim = int(x_lag.shape[1])
            out_dim = int(loaded.y.shape[1])
            model = MLP(in_dim, list(args.hidden),
                        out_dim, float(args.dropout))
            n_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
            print(
                "mlp_lags device",
                device,
                "in_dim",
                in_dim,
                "out_dim",
                out_dim,
                "hidden",
                list(args.hidden),
                "params",
                n_params,
            )
            if args.log_test_mae_norm_each_epoch and show_progress:
                print("diagnostic: test_mae_norm each epoch (monitor only)")

            t0 = time.perf_counter()
            last_ep, best_val = train_mlp(
                model=model,
                x_train=x_train_t,
                y_train=y_train_t,
                x_val=x_val_t,
                y_val=y_val_t,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                show_progress=show_progress,
                early_stopping=use_es,
                overfit_gap_threshold=float(args.overfit_gap_threshold),
                overfit_patience=int(args.overfit_patience),
                x_test_monitor=x_test_mon_t,
                y_test_monitor=y_test_mon_t,
                log_test_mae_norm_each_epoch=bool(
                    args.log_test_mae_norm_each_epoch),
            )
            print(
                "mlp_lags train_wall_seconds",
                round(time.perf_counter() - t0, 3),
                "stopped_epoch",
                last_ep,
                "best_val_mae_norm_macro",
                round(best_val, 6),
            )

            model.eval()
            preds: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, x_test_n.shape[0], args.batch_size):
                    batch = torch.from_numpy(
                        x_test_n[start: start + args.batch_size]).to(device)
                    preds.append(model(batch).cpu().numpy())
            y_pred_n = np.concatenate(preds, axis=0)
            if args.no_standardize_y:
                y_pred = y_pred_n.astype(np.float32)
            else:
                y_pred = denormalize_y(y_pred_n, y_mean, y_std)

            tr_parts: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, x_tr_n.shape[0], args.batch_size):
                    batch = torch.from_numpy(
                        x_tr_n[start: start + args.batch_size]).to(device)
                    tr_parts.append(model(batch).cpu().numpy())
            y_train_pred_n = np.concatenate(tr_parts, axis=0)
            if args.no_standardize_y:
                y_train_pred = y_train_pred_n.astype(np.float32)
            else:
                y_train_pred = denormalize_y(y_train_pred_n, y_mean, y_std)
            train_metrics_summary = print_regression_block(
                "MLP+lags Regression metrics ON TRAIN (fit set, denormalized Y)",
                loaded.y_columns,
                loaded.y[tune_train_idx],
                y_train_pred,
                False,
            )

            if val_idx:
                y_val_pred_n = []
                with torch.inference_mode():
                    for start in range(0, xv.shape[0], args.batch_size):
                        batch = torch.from_numpy(
                            xv[start: start + args.batch_size]).to(device)
                        y_val_pred_n.append(model(batch).cpu().numpy())
                y_val_pred_n = np.concatenate(y_val_pred_n, axis=0)
                if args.no_standardize_y:
                    y_val_pred = y_val_pred_n.astype(np.float32)
                else:
                    y_val_pred = denormalize_y(y_val_pred_n, y_mean, y_std)
                val_metrics_summary = print_regression_block(
                    "MLP+lags metrics ON VALIDATION (denormalized Y)",
                    loaded.y_columns,
                    loaded.y[val_idx],
                    y_val_pred,
                    False,
                )
            else:
                val_metrics_summary = None

            test_metrics = print_regression_block(
                "MLP+lags Regression metrics ON TEST ONLY",
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
                        "model": "mlp_torch_lag_emb",
                        "embedding_lags": str(int(args.embedding_lags)),
                        "embedding_column": args.embedding_column,
                        "rubert_table": args.rubert_table,
                        "music_table": args.music_table,
                        "y_group": args.y_group,
                        "split": args.split,
                        "seed": str(args.seed),
                        "test_fraction": str(args.test_fraction),
                        "test_games": str(args.test_games),
                        "test_rows": str(len(test_idx)),
                        "merge_identical_embeddings": str(bool(args.merge_identical_embeddings)),
                        "hidden": ",".join(str(h) for h in args.hidden),
                        "dropout": str(args.dropout),
                        "lr": str(args.lr),
                        "weight_decay": str(args.weight_decay),
                        "epochs_ran": str(last_ep),
                        "best_val_mae_norm_macro": str(round(best_val, 8)),
                        "standardize_x": str(bool(args.standardize_x)),
                        "standardize_y": str(not args.no_standardize_y),
                    },
                )
                print("wrote per-column metrics csv",
                      str(args.per_column_metrics_csv))

            from text_scripts.ml_scripts.retrieval_column_mask import (
                prepare_retrieval_column_indices,
                retrieval_mask_meta,
            )
            r_col_idx, r_mask_detail = prepare_retrieval_column_indices(
                args,
                y_true_val=loaded.y[val_idx] if val_idx else None,
                y_pred_val=y_val_pred if val_idx else None,
                y_true_test=loaded.y[test_idx],
                y_pred_test=y_pred,
            )

            print("")
            print("retrieval hit rates")
            cands = load_candidates_dir(
                args.dataset_dir) if args.dataset_dir is not None else None
            if cands is not None:
                retrieval = retrieval_hit_rates_from_candidates(
                    cands,
                    [loaded.rows[i] for i in test_idx],
                    y_pred,
                    args.top_percent,
                    r_col_idx,
                )
            elif conn is not None:
                retrieval = retrieval_hit_rates(
                    conn,
                    args.music_table,
                    loaded.y_columns,
                    [loaded.rows[i] for i in test_idx],
                    y_pred,
                    args.top_percent,
                    r_col_idx,
                )
            else:
                retrieval = {m: (0, 0, 0) for m in ("rmse", "mae", "cosine")}
                print("retrieval skipped (no candidates/ on dataset-dir)")
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

            if args.run_dir is not None:
                write_metrics_json(
                    args.run_dir,
                    {
                        "script": "mlp_lags",
                        "train_metrics": train_metrics_summary,
                        "val_metrics": val_metrics_summary,
                        "test_metrics": test_metrics,
                        "retrieval": {
                            k: {"hits": a, "totals": b, "skipped": c}
                            for k, (a, b, c) in retrieval.items()
                        },
                        **retrieval_mask_meta(args, r_col_idx, r_mask_detail),
                        "epochs_ran": int(last_ep),
                        "best_val_mae_norm_macro": float(best_val),
                        "train_wall_seconds": float(time.perf_counter() - t0),
                    },
                )

            if conn is not None and not args.skip_experiment_log:
                from text_scripts.ml_scripts.db_consts import TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE
                from text_scripts.ml_scripts.train_experiment_log import log_mlp_run

                log_mlp_run(
                    conn,
                    argv_text=" ".join(sys.argv),
                    args=args,
                    n_rows_loaded=len(loaded.rows),
                    n_y_columns=len(loaded.y_columns),
                    n_train_fit=len(tune_train_idx),
                    n_val=len(val_idx),
                    n_test=len(test_idx),
                    train_metrics=train_metrics_summary,
                    val_metrics=val_metrics_summary,
                    test_metrics=test_metrics,
                    retrieval=retrieval,
                    train_wall_s=float(time.perf_counter() - t0),
                    epochs_ran=int(last_ep),
                    best_val_mae_norm=float(best_val),
                    script="mlp_lags",
                )
                print("logged experiment to", TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
