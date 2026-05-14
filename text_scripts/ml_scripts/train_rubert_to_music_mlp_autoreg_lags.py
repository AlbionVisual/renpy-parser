from __future__ import annotations
import argparse
import math
import secrets
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

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
from text_scripts.ml_scripts.mlp_train_defaults import (
    register_autoreg_lags_only_cli,
    register_shared_rubert_mlp_cli,
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
    fetch_candidates,
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
from text_scripts.ml_scripts.train_rubert_to_music_mlp import (
    MLP,
    denormalize_y,
    parse_hidden,
    pick_device,
    standardize,
)
from text_scripts.ml_scripts.train_rubert_to_music_mlp_lags import build_concat_embedding_lags


def nearest_music_id_mae(
    y_raw: np.ndarray,
    candidates: list[tuple[int, np.ndarray]],
) -> int | None:
    if len(candidates) <= 1:
        return None
    mat = np.stack([vec for _mid, vec in candidates],
                   axis=0).astype(np.float64)
    dist = np.mean(np.abs(mat - y_raw.astype(np.float64)), axis=1)
    j = int(np.argmin(dist))
    return candidates[j][0]


class PerGameAutoregState:
    __slots__ = ("y_dim", "n_pred", "slots", "prev_nn", "streak")

    def __init__(self, y_dim: int, n_pred: int) -> None:
        self.y_dim = y_dim
        self.n_pred = n_pred
        z = np.zeros(y_dim, dtype=np.float32)
        self.slots = [z.copy() for _ in range(n_pred)] if n_pred > 0 else []
        self.prev_nn: int | None = None
        self.streak = 0

    def suffix(self, streak_scale: float) -> np.ndarray:
        st = np.array([self.streak / float(streak_scale)], dtype=np.float32)
        if self.n_pred <= 0:
            return st
        flat = np.concatenate(self.slots, axis=0).astype(np.float32)
        return np.concatenate([flat, st], axis=0)

    def after_step(
        self,
        y_hat_n: np.ndarray,
        y_true_n_row: np.ndarray,
        *,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        candidates: list[tuple[int, np.ndarray]],
        mixing_prob: float,
        training: bool,
        rng: np.random.Generator,
    ) -> None:
        y_hat_raw = denormalize_y(
            y_hat_n.reshape(1, -1),
            y_mean,
            y_std,
        ).astype(np.float64)[0]
        nn_id = nearest_music_id_mae(y_hat_raw, candidates)
        if nn_id is not None:
            if self.prev_nn is not None and nn_id == self.prev_nn:
                self.streak += 1
            else:
                self.streak = 0
            self.prev_nn = nn_id
        use_pred = training and mixing_prob > 0.0 and float(
            rng.random()) < mixing_prob
        if use_pred:
            v = y_hat_n.astype(np.float32).copy()
        else:
            v = y_true_n_row.astype(np.float32).copy()
        if self.n_pred > 0:
            self.slots = [v] + self.slots[:-1]


def mixing_probability(
    epoch_1: int,
    epochs: int,
    warmup: int,
    p_final: float,
    linear_ramp: bool,
) -> float:
    if warmup >= epochs:
        return 0.0
    if epoch_1 <= warmup:
        return 0.0
    if not linear_ramp:
        return float(p_final)
    span = max(1, epochs - warmup)
    t = (epoch_1 - warmup) / float(span)
    return float(p_final * min(1.0, t))


def sort_indices_by_game_phrase(rows: Sequence[PairRow], idxs: Sequence[int]) -> list[int]:
    return sorted(
        idxs,
        key=lambda i: (rows[i].game, rows[i].phrase_order, rows[i].music_id),
    )


def shuffle_train_order_by_game(
    rows: Sequence[PairRow],
    idxs: Sequence[int],
    rng: np.random.Generator,
) -> list[int]:
    by_game: dict[str, list[int]] = defaultdict(list)
    for i in idxs:
        by_game[rows[i].game].append(i)
    for g in list(by_game.keys()):
        by_game[g].sort(key=lambda i: (rows[i].phrase_order, rows[i].music_id))
    games = list(by_game.keys())
    rng.shuffle(games)
    out: list[int] = []
    for g in games:
        out.extend(by_game[g])
    return out


def sequential_mae_norm_macro(
    model: nn.Module,
    ordered_indices: list[int],
    rows: Sequence[PairRow],
    x_lag: np.ndarray,
    y_full_n: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    candidates_by_game: dict[str, list[tuple[int, np.ndarray]]],
    n_pred_lags: int,
    streak_scale: float,
    device: torch.device,
) -> float:
    model.eval()
    y_dim = int(y_full_n.shape[1])
    abs_sum = 0.0
    n_seen = 0
    current_game: str | None = None
    st: PerGameAutoregState | None = None
    with torch.inference_mode():
        for i in ordered_indices:
            g = rows[i].game
            if g != current_game:
                current_game = g
                st = PerGameAutoregState(y_dim, n_pred_lags)
            assert st is not None
            xb = np.concatenate(
                [x_lag[i].astype(np.float32), st.suffix(streak_scale)],
                axis=0,
            )
            x_in = torch.from_numpy(xb).unsqueeze(0).to(device)
            pred = model(x_in).cpu().numpy().astype(np.float64)[0]
            tgt = y_full_n[i].astype(np.float64)
            abs_sum += float(np.mean(np.abs(pred - tgt)))
            n_seen += 1
            cand = candidates_by_game.get(g, [])
            st.after_step(
                pred.astype(np.float32),
                y_full_n[i].astype(np.float32),
                y_mean=y_mean,
                y_std=y_std,
                candidates=cand,
                mixing_prob=0.0,
                training=False,
                rng=np.random.default_rng(0),
            )
    return abs_sum / max(1, n_seen)


def train_autoreg_sequential(
    *,
    model: nn.Module,
    rows: Sequence[PairRow],
    train_order_fn,
    x_lag: np.ndarray,
    y_full_n: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    candidates_by_game: dict[str, list[tuple[int, np.ndarray]]],
    n_pred_lags: int,
    streak_scale: float,
    epochs: int,
    sample_warmup_epochs: int,
    sample_p_final: float,
    sample_linear_ramp: bool,
    base_seed: int,
    device: torch.device,
    val_ordered: list[int],
    lr: float,
    weight_decay: float,
    patience: int,
    early_stopping: bool,
    overfit_gap_threshold: float,
    overfit_patience: int,
    show_progress: bool,
    test_ordered: list[int],
    log_test_each_epoch: bool,
) -> tuple[int, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.to(device)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0
    bad_gap = 0
    last_ep = 0
    use_gap = overfit_gap_threshold > 0.0 and len(val_ordered) > 0
    y_dim = int(y_full_n.shape[1])

    for ep in range(1, epochs + 1):
        p_mix = mixing_probability(
            ep,
            epochs,
            sample_warmup_epochs,
            sample_p_final,
            sample_linear_ramp,
        )
        rng_ep = np.random.default_rng(int(base_seed) + ep * 1_000_003)
        if show_progress and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print("scheduled_sampling_p_use_pred_hat", round(p_mix, 6))
        model.train()
        order = train_order_fn(rng_ep)
        train_loss = 0.0
        n_steps = 0
        current_game: str | None = None
        st: PerGameAutoregState | None = None
        for i in order:
            g = rows[i].game
            if g != current_game:
                current_game = g
                st = PerGameAutoregState(y_dim, n_pred_lags)
            assert st is not None
            xb = np.concatenate(
                [x_lag[i].astype(np.float32), st.suffix(streak_scale)],
                axis=0,
            )
            x_in = torch.from_numpy(xb).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_full_n[i: i + 1]).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x_in)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach().cpu().item())
            n_steps += 1
            pred_np = pred.detach().cpu().numpy().astype(np.float32)[0]
            st.after_step(
                pred_np,
                y_full_n[i].astype(np.float32),
                y_mean=y_mean,
                y_std=y_std,
                candidates=candidates_by_game.get(g, []),
                mixing_prob=p_mix,
                training=True,
                rng=rng_ep,
            )
        train_mse = train_loss / max(1, n_steps)
        vm = float("nan")
        tm = float("nan")
        gap = float("nan")
        if len(val_ordered) > 0:
            vm = sequential_mae_norm_macro(
                model,
                val_ordered,
                rows,
                x_lag,
                y_full_n,
                y_mean,
                y_std,
                candidates_by_game,
                n_pred_lags,
                streak_scale,
                device,
            )
        if n_steps > 0:
            tm = sequential_mae_norm_macro(
                model,
                order,
                rows,
                x_lag,
                y_full_n,
                y_mean,
                y_std,
                candidates_by_game,
                n_pred_lags,
                streak_scale,
                device,
            )
        if not math.isnan(vm) and not math.isnan(tm):
            gap = vm - tm
        if early_stopping and len(val_ordered) > 0:
            if vm < best_val - 1e-8:
                best_val = vm
                bad = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
        else:
            if len(val_ordered) > 0:
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_val = vm if not math.isnan(vm) else best_val
            else:
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            bad = 0
        if use_gap and not math.isnan(gap) and gap > overfit_gap_threshold:
            bad_gap += 1
        else:
            bad_gap = 0
        last_ep = ep
        test_mae = float("nan")
        if log_test_each_epoch and len(test_ordered) > 0:
            test_mae = sequential_mae_norm_macro(
                model,
                test_ordered,
                rows,
                x_lag,
                y_full_n,
                y_mean,
                y_std,
                candidates_by_game,
                n_pred_lags,
                streak_scale,
                device,
            )
        want_line = show_progress and (
            log_test_each_epoch
            or ep == 1
            or ep % 5 == 0
            or ep == epochs
            or (early_stopping and bad == 0)
            or (use_gap and bad_gap == 1)
        )
        if want_line:
            if log_test_each_epoch:
                print(
                    "epoch",
                    ep,
                    "train_mse_seq",
                    round(train_mse, 6),
                    "train_mae_norm_seq",
                    round(tm, 6) if not math.isnan(tm) else "nan",
                    "val_mae_norm_seq",
                    round(vm, 6) if not math.isnan(vm) else "nan",
                    "test_mae_norm_seq_monitor",
                    round(test_mae, 6) if not math.isnan(test_mae) else "nan",
                    "val_minus_train_mae_norm",
                    round(gap, 6) if not math.isnan(gap) else "nan",
                    "best_val_mae_norm_seq",
                    round(best_val, 6) if best_val < float("inf") else "nan",
                    "patience_val",
                    bad,
                    "/",
                    patience if early_stopping else "off",
                    "patience_overfit_gap",
                    bad_gap,
                    "/",
                    overfit_patience if use_gap else "off",
                )
            else:
                print(
                    "epoch",
                    ep,
                    "train_mse_seq",
                    round(train_mse, 6),
                    "train_mae_norm_seq",
                    round(tm, 6) if not math.isnan(tm) else "nan",
                    "val_mae_norm_seq",
                    round(vm, 6) if not math.isnan(vm) else "nan",
                    "val_minus_train_mae_norm",
                    round(gap, 6) if not math.isnan(gap) else "nan",
                    "best_val_mae_norm_seq",
                    round(best_val, 6) if best_val < float("inf") else "nan",
                    "patience_val",
                    bad,
                    "/",
                    patience if early_stopping else "off",
                    "patience_overfit_gap",
                    bad_gap,
                    "/",
                    overfit_patience if use_gap else "off",
                )
        if early_stopping and bad >= patience:
            if show_progress:
                print("early stop: val metric patience exhausted")
            break
        if use_gap and bad_gap >= overfit_patience:
            if show_progress:
                print(
                    "early stop: val_minus_train_mae_norm >",
                    overfit_gap_threshold,
                    "for",
                    overfit_patience,
                    "epochs",
                )
            break
    model.load_state_dict(best_state)
    return last_ep, best_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLP+lags+autoreg: RuBERT lags + prev preds + streak -> music_data.",
    )
    register_shared_rubert_mlp_cli(
        p,
        parse_hidden=parse_hidden,
        device_default="cpu",
        include_embedding_lags=True,
    )
    register_autoreg_lags_only_cli(p)
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
    if args.pred_lags < 0:
        print("--pred-lags must be >= 0")
        return 2
    if args.sample_warmup_epochs < 0:
        print("--sample-warmup-epochs must be >= 0")
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
            extra_in = int(args.pred_lags) * int(loaded.y.shape[1]) + 1
            print(
                "mlp_autoreg_lags raw_emb_dim",
                loaded.x.shape[1],
                "embedding_lags",
                int(args.embedding_lags),
                "concat_in_dim",
                x_lag.shape[1],
                "pred_lags",
                int(args.pred_lags),
                "extra_in",
                extra_in,
                "rows",
                len(loaded.rows),
            )

            held_out_games: list[str] | None = None
            if args.split == "within_game":
                train_idx, test_idx = split_within_game(
                    loaded.rows, args.test_fraction, args.seed
                )
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
            print_split_visibility(
                loaded.rows, train_idx, test_idx, args, held_out_games)

            if args.val_split == "game":
                tune_train_idx, val_idx = split_validation_games_held_out_from_train(
                    loaded.rows,
                    train_idx,
                    args.val_games,
                    args.seed + 1,
                )
                if not val_idx:
                    print("warning: val-split game failed; no val")
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
                "mlp_autoreg_lags val_split",
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
                "mlp_autoreg_lags index_overlap tune_val",
                len(st & sv),
                "tune_test",
                len(st & ste),
                "val_test",
                len(sv & ste),
            )
            if st & sv or st & ste or sv & ste:
                print("fatal: index overlap")
                return 2

            y_tr = loaded.y[tune_train_idx]
            if args.no_standardize_y:
                y_mean = np.zeros(loaded.y.shape[1], dtype=np.float32)
                y_std = np.ones(loaded.y.shape[1], dtype=np.float32)
            else:
                _, y_mean, y_std = standardize(y_tr)

            y_full = loaded.y.astype(np.float64)
            if args.no_standardize_y:
                y_full_n = y_full.astype(np.float32)
            else:
                y_full_n = ((y_full - y_mean) / y_std).astype(np.float32)

            if args.standardize_x:
                x_tr = x_lag[tune_train_idx]
                x_tr_n, x_fit_mean, x_fit_std = standardize(x_tr)
            else:
                x_tr_n = x_lag.astype(np.float32)
                x_fit_mean = x_fit_std = None

            if x_fit_mean is not None:
                x_lag_n = (
                    (x_lag.astype(np.float64) - x_fit_mean) / x_fit_std
                ).astype(np.float32)
            else:
                x_lag_n = x_lag.astype(np.float32)

            games_union = sorted(
                {loaded.rows[i].game for i in tune_train_idx}
                | {loaded.rows[i].game for i in val_idx}
                | {loaded.rows[i].game for i in test_idx}
            )
            if args.dataset_dir is not None:
                disk_c = load_candidates_dir(args.dataset_dir)
                if disk_c is None:
                    print("fatal: mlp_autoreg_lags needs candidates/ in dataset-dir")
                    return 2
                candidates_by_game = {
                    g: list(disk_c.get(g, [])) for g in games_union}
            else:
                candidates_by_game = fetch_candidates(
                    conn,
                    args.music_table,
                    loaded.y_columns,
                    games_union,
                )

            val_ordered = sort_indices_by_game_phrase(loaded.rows, val_idx)
            test_ordered = sort_indices_by_game_phrase(loaded.rows, test_idx)

            in_dim = int(x_lag_n.shape[1]) + extra_in
            out_dim = int(loaded.y.shape[1])
            model = MLP(in_dim, list(args.hidden),
                        out_dim, float(args.dropout))
            n_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
            print(
                "mlp_autoreg_lags device",
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
                print("diagnostic: sequential test_mae_norm each epoch (monitor only)")

            def train_order_fn(rng: np.random.Generator) -> list[int]:
                return shuffle_train_order_by_game(loaded.rows, tune_train_idx, rng)

            t0 = time.perf_counter()
            last_ep, best_val = train_autoreg_sequential(
                model=model,
                rows=loaded.rows,
                train_order_fn=train_order_fn,
                x_lag=x_lag_n,
                y_full_n=y_full_n,
                y_mean=y_mean,
                y_std=y_std,
                candidates_by_game=candidates_by_game,
                n_pred_lags=int(args.pred_lags),
                streak_scale=float(args.streak_scale),
                epochs=int(args.epochs),
                sample_warmup_epochs=int(args.sample_warmup_epochs),
                sample_p_final=float(args.sample_p_final),
                sample_linear_ramp=not bool(args.no_sample_linear_ramp),
                base_seed=int(args.seed),
                device=device,
                val_ordered=val_ordered,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                patience=args.patience,
                early_stopping=bool(val_idx),
                overfit_gap_threshold=float(args.overfit_gap_threshold),
                overfit_patience=int(args.overfit_patience),
                show_progress=show_progress,
                test_ordered=test_ordered,
                log_test_each_epoch=bool(args.log_test_mae_norm_each_epoch),
            )
            print(
                "mlp_autoreg_lags train_wall_seconds",
                round(time.perf_counter() - t0, 3),
                "stopped_epoch",
                last_ep,
                "best_val_mae_norm_seq",
                round(best_val, 6),
            )

            def collect_preds(ordered: list[int]) -> tuple[np.ndarray, list[int]]:
                model.eval()
                preds: list[np.ndarray] = []
                y_dim = int(loaded.y.shape[1])
                current_game: str | None = None
                st: PerGameAutoregState | None = None
                with torch.inference_mode():
                    for i in ordered:
                        g = loaded.rows[i].game
                        if g != current_game:
                            current_game = g
                            st = PerGameAutoregState(
                                y_dim, int(args.pred_lags))
                        assert st is not None
                        xb = np.concatenate(
                            [x_lag_n[i].astype(np.float32), st.suffix(
                                float(args.streak_scale))],
                            axis=0,
                        )
                        x_in = torch.from_numpy(xb).unsqueeze(0).to(device)
                        pred = model(x_in).cpu().numpy().astype(np.float32)[0]
                        preds.append(pred)
                        st.after_step(
                            pred,
                            y_full_n[i].astype(np.float32),
                            y_mean=y_mean,
                            y_std=y_std,
                            candidates=candidates_by_game.get(g, []),
                            mixing_prob=0.0,
                            training=False,
                            rng=np.random.default_rng(0),
                        )
                return np.stack(preds, axis=0), ordered

            y_test_pred_n, ord_test = collect_preds(test_ordered)
            inv_map = {j: k for k, j in enumerate(ord_test)}
            y_pred_n_full = np.zeros(
                (len(test_idx), out_dim), dtype=np.float32)
            for u, orig_i in enumerate(test_idx):
                pos = inv_map[orig_i]
                y_pred_n_full[u] = y_test_pred_n[pos]
            if args.no_standardize_y:
                y_pred = y_pred_n_full.astype(np.float32)
            else:
                y_pred = denormalize_y(y_pred_n_full, y_mean, y_std)

            y_train_pred_n, ord_tr = collect_preds(
                shuffle_train_order_by_game(
                    loaded.rows,
                    tune_train_idx,
                    np.random.default_rng(args.seed),
                )
            )
            inv_tr = {j: k for k, j in enumerate(ord_tr)}
            y_tr_pred_n = np.zeros(
                (len(tune_train_idx), out_dim), dtype=np.float32)
            for u, orig_i in enumerate(tune_train_idx):
                y_tr_pred_n[u] = y_train_pred_n[inv_tr[orig_i]]
            if args.no_standardize_y:
                y_train_pred = y_tr_pred_n.astype(np.float32)
            else:
                y_train_pred = denormalize_y(y_tr_pred_n, y_mean, y_std)

            train_metrics_summary = print_regression_block(
                "MLP+lags+autoreg Regression metrics ON TRAIN (fit set, denormalized Y; order shuffled once for eval)",
                loaded.y_columns,
                loaded.y[tune_train_idx],
                y_train_pred,
                False,
            )

            val_metrics_summary = None
            if val_idx:
                y_val_pred_n, ord_v = collect_preds(val_ordered)
                inv_v = {j: k for k, j in enumerate(ord_v)}
                y_va_n = np.zeros((len(val_idx), out_dim), dtype=np.float32)
                for u, orig_i in enumerate(val_idx):
                    y_va_n[u] = y_val_pred_n[inv_v[orig_i]]
                if args.no_standardize_y:
                    y_val_pred = y_va_n.astype(np.float32)
                else:
                    y_val_pred = denormalize_y(y_va_n, y_mean, y_std)
                val_metrics_summary = print_regression_block(
                    "MLP+lags+autoreg metrics ON VALIDATION (denormalized Y, sequential)",
                    loaded.y_columns,
                    loaded.y[val_idx],
                    y_val_pred,
                    False,
                )

            test_metrics = print_regression_block(
                "MLP+lags+autoreg Regression metrics ON TEST ONLY (sequential)",
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
                        "model": "mlp_torch_autoreg_lags",
                        "embedding_lags": str(int(args.embedding_lags)),
                        "pred_lags": str(int(args.pred_lags)),
                        "streak_scale": str(float(args.streak_scale)),
                        "sample_warmup_epochs": str(int(args.sample_warmup_epochs)),
                        "sample_p_final": str(float(args.sample_p_final)),
                        "sample_linear_ramp": str(not bool(args.no_sample_linear_ramp)),
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
                        "script": "mlp_autoreg_lags",
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
                    script="mlp_autoreg_lags",
                )
                print("logged experiment to", TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
