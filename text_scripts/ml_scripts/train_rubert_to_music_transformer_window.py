from __future__ import annotations
import argparse
import math
import secrets
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
from text_scripts.ml_scripts.mlp_train_defaults import register_shared_rubert_mlp_cli
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
from text_scripts.ml_scripts.train_rubert_to_music_mlp import parse_hidden, pick_device, standardize
from text_scripts.ml_scripts.train_rubert_to_music_mlp_lags import build_concat_embedding_lags


class WindowTransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        dropout: float,
        window: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.emb = nn.Linear(in_dim, int(d_model))
        enc_layer = nn.TransformerEncoderLayer(
            int(d_model),
            int(nhead),
            int(dim_ff),
            float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, int(nlayers))
        self.head = nn.Linear(int(d_model), out_dim)

    def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_padded.shape
        device = x_padded.device
        h = self.emb(x_padded)
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        kpm = idx >= lengths.to(device).unsqueeze(1)
        m = torch.zeros(T, T, dtype=h.dtype, device=device)
        w = self.window
        for i in range(T):
            lo = max(0, i - w + 1)
            if lo > 0:
                m[i, :lo] = float("-inf")
            m[i, i + 1:] = float("-inf")
        z = self.enc(h, mask=m, src_key_padding_mask=kpm)
        finite_z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        valid_bt = (
            idx < lengths.to(device).unsqueeze(1)
        ).to(dtype=finite_z.dtype)
        masked = finite_z * valid_bt.unsqueeze(-1)
        return self.head(masked)


def _by_game_ordered_indices(rows: Sequence[PairRow], idxs: Sequence[int]) -> dict[str, list[int]]:
    by_game: dict[str, list[int]] = defaultdict(list)
    for i in idxs:
        by_game[rows[i].game].append(i)
    for g in list(by_game.keys()):
        by_game[g].sort(key=lambda j: (rows[j].phrase_order, rows[j].music_id))
    return by_game


class GameSeqDataset(Dataset):
    def __init__(
        self,
        *,
        rows: Sequence[PairRow],
        x_all: np.ndarray,
        y_all: np.ndarray,
        game_to_indices: dict[str, list[int]],
    ) -> None:
        self.rows = rows
        self.x_all = x_all
        self.y_all = y_all
        self.games = sorted(game_to_indices.keys())
        self.game_to_indices = game_to_indices

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        g = self.games[idx]
        inds = self.game_to_indices[g]
        x = self.x_all[inds].astype(np.float32, copy=False)
        y = self.y_all[inds].astype(np.float32, copy=False)
        return g, x, y, np.asarray(inds, dtype=np.int64)


def collate_game_seqs(
    batch: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
    games = [g for g, _x, _y, _idx in batch]
    lengths = torch.tensor([int(x.shape[0])
                           for _g, x, _y, _idx in batch], dtype=torch.int64)
    max_len = int(torch.max(lengths).item()) if len(batch) else 0
    x_dim = int(batch[0][1].shape[1]) if len(batch) else 0
    y_dim = int(batch[0][2].shape[1]) if len(batch) else 0
    xb = torch.zeros((len(batch), max_len, x_dim), dtype=torch.float32)
    yb = torch.zeros((len(batch), max_len, y_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    indices: list[np.ndarray] = []
    for i, (_g, x, y, idxs) in enumerate(batch):
        n = int(x.shape[0])
        xb[i, :n] = torch.from_numpy(x)
        yb[i, :n] = torch.from_numpy(y)
        mask[i, :n] = True
        indices.append(idxs)
    return games, xb, yb, lengths, mask, indices


def masked_mae_macro(y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (y_hat - y_true).abs()
    m = mask.unsqueeze(-1).to(diff.dtype)
    denom = torch.clamp(m.sum(dim=(0, 1)), min=1.0)
    per_col = (diff * m).sum(dim=(0, 1)) / denom
    return float(per_col.mean().detach().cpu().item())


def train_tf_window(
    *,
    model: WindowTransformerRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    patience: int,
    overfit_gap_threshold: float,
    overfit_patience: int,
    show_progress: bool,
    log_test_each_epoch: bool,
    test_loader: DataLoader | None,
) -> tuple[int, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction="none")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0
    bad_gap = 0
    last_ep = 0
    use_gap = overfit_gap_threshold > 0.0 and val_loader is not None

    def eval_loader(loader: DataLoader) -> float:
        model.eval()
        total = 0.0
        n_batches = 0
        with torch.inference_mode():
            for _games, xb, yb, lengths, mask, _idxs in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                pred = model(xb, lengths)
                total += masked_mae_macro(pred, yb, mask)
                n_batches += 1
        return total / max(1, n_batches)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for _games, xb, yb, lengths, mask, _idxs in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, lengths)
            per = loss_fn(pred, yb).mean(dim=2)
            m = mask.to(per.dtype)
            loss = (per * m).sum() / torch.clamp(m.sum(), min=1.0)
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            opt.step()
            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
        train_mse = total_loss / max(1, n_batches)

        tm = float("nan")
        vm = float("nan")
        gap = float("nan")
        if train_loader is not None:
            tm = eval_loader(train_loader)
        if val_loader is not None:
            vm = eval_loader(val_loader)
        if not math.isnan(vm) and not math.isnan(tm):
            gap = vm - tm

        if val_loader is not None:
            if vm < best_val - 1e-8:
                best_val = vm
                bad = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
        else:
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_val = vm if not math.isnan(vm) else best_val
            bad = 0

        if use_gap and not math.isnan(gap) and gap > overfit_gap_threshold:
            bad_gap += 1
        else:
            bad_gap = 0

        last_ep = ep
        test_mae = float("nan")
        if log_test_each_epoch and test_loader is not None:
            test_mae = eval_loader(test_loader)

        want_line = show_progress and (
            log_test_each_epoch
            or ep == 1
            or ep % 5 == 0
            or ep == epochs
            or (val_loader is not None and bad == 0)
            or (use_gap and bad_gap == 1)
        )
        if want_line:
            if log_test_each_epoch:
                print(
                    "epoch",
                    ep,
                    "train_mse",
                    round(train_mse, 6),
                    "train_mae_norm_macro",
                    round(tm, 6) if not math.isnan(tm) else "nan",
                    "val_mae_norm_macro",
                    round(vm, 6) if not math.isnan(vm) else "nan",
                    "test_mae_norm_monitor_only",
                    round(test_mae, 6) if not math.isnan(test_mae) else "nan",
                    "val_minus_train_mae_norm",
                    round(gap, 6) if not math.isnan(gap) else "nan",
                    "best_val_mae_norm_macro",
                    round(best_val, 6) if best_val < float("inf") else "nan",
                    "patience_val",
                    bad,
                    "/",
                    patience if val_loader is not None else "off",
                    "patience_overfit_gap",
                    bad_gap,
                    "/",
                    overfit_patience if use_gap else "off",
                )
            else:
                print(
                    "epoch",
                    ep,
                    "train_mse",
                    round(train_mse, 6),
                    "train_mae_norm_macro",
                    round(tm, 6) if not math.isnan(tm) else "nan",
                    "val_mae_norm_macro",
                    round(vm, 6) if not math.isnan(vm) else "nan",
                    "val_minus_train_mae_norm",
                    round(gap, 6) if not math.isnan(gap) else "nan",
                    "best_val_mae_norm_macro",
                    round(best_val, 6) if best_val < float("inf") else "nan",
                    "patience_val",
                    bad,
                    "/",
                    patience if val_loader is not None else "off",
                    "patience_overfit_gap",
                    bad_gap,
                    "/",
                    overfit_patience if use_gap else "off",
                )

        if val_loader is not None and bad >= patience:
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
                    "epochs (hint: train memorization vs val generalization)",
                )
            break

    model.load_state_dict(best_state)
    return last_ep, best_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Windowed causal Transformer: RuBERT -> music_data, sequence per game.",
    )
    register_shared_rubert_mlp_cli(
        p,
        parse_hidden=parse_hidden,
        device_default="cpu",
        include_embedding_lags=True,
    )
    p.add_argument("--tf-window", type=int, default=512)
    p.add_argument("--tf-d-model", type=int, default=256)
    p.add_argument("--tf-layers", type=int, default=4)
    p.add_argument("--tf-heads", type=int, default=4)
    p.add_argument("--tf-dim-ff", type=int, default=1024)
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="L2 gradient clip after each train step (0 disables)",
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
    if args.tf_window < 1:
        print("--tf-window must be >= 1")
        return 2
    if args.tf_d_model < 1:
        print("--tf-d-model must be >= 1")
        return 2
    if args.tf_d_model % args.tf_heads != 0:
        print("--tf-d-model must be divisible by --tf-heads")
        return 2
    if args.tf_layers < 1:
        print("--tf-layers must be >= 1")
        return 2
    if args.tf_heads < 1:
        print("--tf-heads must be >= 1")
        return 2
    if args.tf_dim_ff < 1:
        print("--tf-dim-ff must be >= 1")
        return 2
    if args.grad_clip_norm < 0:
        print("--grad-clip-norm must be >= 0")
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
                "tf_window raw_emb_dim",
                loaded.x.shape[1],
                "embedding_lags",
                int(args.embedding_lags),
                "in_dim",
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
                "tf_window val_split",
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
                "tf_window index_overlap tune_val",
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
            if args.standardize_x:
                x_tr_n, x_mean, x_std = standardize(x_tr)
                x_fit_mean, x_fit_std = x_mean, x_std
            else:
                x_fit_mean = x_fit_std = None

            if x_fit_mean is not None:
                x_all_n = ((x_lag.astype(np.float64) - x_fit_mean) /
                           x_fit_std).astype(np.float32)
            else:
                x_all_n = x_lag.astype(np.float32)

            y_tr = loaded.y[tune_train_idx]
            if args.no_standardize_y:
                y_mean = np.zeros(y_tr.shape[1], dtype=np.float32)
                y_std = np.ones(y_tr.shape[1], dtype=np.float32)
                y_all_n = loaded.y.astype(np.float32)
            else:
                _y_tr_n, y_mean, y_std = standardize(y_tr)
                y_all_n = ((loaded.y.astype(np.float64) - y_mean) /
                           y_std).astype(np.float32)

            game_train = _by_game_ordered_indices(loaded.rows, tune_train_idx)
            game_val = _by_game_ordered_indices(
                loaded.rows, val_idx) if val_idx else {}
            game_test = _by_game_ordered_indices(loaded.rows, test_idx)

            ds_train = GameSeqDataset(
                rows=loaded.rows, x_all=x_all_n, y_all=y_all_n, game_to_indices=game_train)
            ds_val = GameSeqDataset(rows=loaded.rows, x_all=x_all_n,
                                    y_all=y_all_n, game_to_indices=game_val) if val_idx else None
            ds_test = GameSeqDataset(
                rows=loaded.rows, x_all=x_all_n, y_all=y_all_n, game_to_indices=game_test)

            train_loader = DataLoader(
                ds_train,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=collate_game_seqs,
            )
            val_loader = (
                DataLoader(
                    ds_val,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_game_seqs,
                )
                if ds_val is not None
                else None
            )
            test_loader = DataLoader(
                ds_test,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_game_seqs,
            )

            in_dim = int(x_all_n.shape[1])
            out_dim = int(loaded.y.shape[1])
            model = WindowTransformerRegressor(
                in_dim=in_dim,
                d_model=int(args.tf_d_model),
                nhead=int(args.tf_heads),
                nlayers=int(args.tf_layers),
                dim_ff=int(args.tf_dim_ff),
                dropout=float(args.dropout),
                window=int(args.tf_window),
                out_dim=out_dim,
            )
            n_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
            print(
                "tf_window device",
                device,
                "in_dim",
                in_dim,
                "out_dim",
                out_dim,
                "tf_window",
                int(args.tf_window),
                "tf_d_model",
                int(args.tf_d_model),
                "tf_layers",
                int(args.tf_layers),
                "tf_heads",
                int(args.tf_heads),
                "tf_dim_ff",
                int(args.tf_dim_ff),
                "params",
                n_params,
            )
            if args.log_test_mae_norm_each_epoch and show_progress:
                print("diagnostic: test MAE (norm. Y) each epoch (monitor only)")

            t0 = time.perf_counter()
            last_ep, best_val = train_tf_window(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=float(args.grad_clip_norm),
                patience=args.patience,
                overfit_gap_threshold=float(args.overfit_gap_threshold),
                overfit_patience=int(args.overfit_patience),
                show_progress=show_progress,
                log_test_each_epoch=bool(args.log_test_mae_norm_each_epoch),
                test_loader=test_loader,
            )
            print(
                "tf_window train_wall_seconds",
                round(time.perf_counter() - t0, 3),
                "stopped_epoch",
                last_ep,
                "best_val_mae_norm_macro",
                round(best_val, 6),
            )

            def predict_all_rows(loader: DataLoader) -> dict[int, np.ndarray]:
                model.eval()
                out: dict[int, np.ndarray] = {}
                with torch.inference_mode():
                    for _games, xb, _yb, lengths, mask, indices in loader:
                        xb = xb.to(device)
                        lengths = lengths.to(device)
                        pred_t = model(xb, lengths).detach().cpu()
                        if not torch.isfinite(pred_t).all():
                            bad = int((~torch.isfinite(pred_t)).sum().item())
                            warnings.warn(
                                "transformer_window predict: non-finite values in raw pred (count=%d); zeroing for metrics"
                                % bad,
                                RuntimeWarning,
                            )
                            pred_t = torch.nan_to_num(
                                pred_t, nan=0.0, posinf=0.0, neginf=0.0
                            )
                        pred = pred_t.numpy().astype(np.float32)
                        mask_np = mask.numpy()
                        for b in range(pred.shape[0]):
                            idxs = indices[b]
                            n = int(mask_np[b].sum())
                            for t in range(n):
                                out[int(idxs[t])] = pred[b, t]
                return out

            pred_map_test = predict_all_rows(test_loader)
            y_pred_n = np.stack([pred_map_test[i]
                                for i in test_idx], axis=0).astype(np.float32)
            if args.no_standardize_y:
                y_pred = y_pred_n.astype(np.float32)
            else:
                y_pred = (y_pred_n.astype(np.float64) *
                          y_std + y_mean).astype(np.float32)

            pred_map_train = predict_all_rows(
                DataLoader(
                    ds_train,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_game_seqs,
                )
            )
            y_train_pred_n = np.stack(
                [pred_map_train[i] for i in tune_train_idx], axis=0).astype(np.float32)
            if args.no_standardize_y:
                y_train_pred = y_train_pred_n.astype(np.float32)
            else:
                y_train_pred = (y_train_pred_n.astype(np.float64)
                                * y_std + y_mean).astype(np.float32)

            train_metrics_summary = print_regression_block(
                "Transformer-window Regression metrics ON TRAIN (fit set, denormalized Y)",
                loaded.y_columns,
                loaded.y[tune_train_idx],
                y_train_pred,
                False,
            )

            if val_idx:
                pred_map_val = predict_all_rows(
                    val_loader) if val_loader is not None else {}
                y_val_pred_n = np.stack(
                    [pred_map_val[i] for i in val_idx], axis=0).astype(np.float32)
                if args.no_standardize_y:
                    y_val_pred = y_val_pred_n.astype(np.float32)
                else:
                    y_val_pred = (y_val_pred_n.astype(np.float64)
                                  * y_std + y_mean).astype(np.float32)
                val_metrics_summary = print_regression_block(
                    "Transformer-window metrics ON VALIDATION (denormalized Y)",
                    loaded.y_columns,
                    loaded.y[val_idx],
                    y_val_pred,
                    False,
                )
            else:
                val_metrics_summary = None

            test_metrics = print_regression_block(
                "Transformer-window Regression metrics ON TEST ONLY",
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
                        "model": "transformer_window_torch",
                        "embedding_lags": str(int(args.embedding_lags)),
                        "tf_window": str(int(args.tf_window)),
                        "tf_d_model": str(int(args.tf_d_model)),
                        "tf_layers": str(int(args.tf_layers)),
                        "tf_heads": str(int(args.tf_heads)),
                        "tf_dim_ff": str(int(args.tf_dim_ff)),
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
                        "script": "transformer_window",
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
                    script="transformer_window",
                )
                print("logged experiment to", TEXT_MUSIC_TRAIN_EXPERIMENTS_TABLE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
