from __future__ import annotations
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

from text_scripts.ml_scripts.db_consts import TEXT_MUSICS_RUBERT_TABLE
from text_scripts.ml_scripts.retrieval_column_mask import (
    register_retrieval_column_mask_cli,
)

DEFAULT_EMBEDDING_COLUMN = "text_emb_concat6_stride4_overlap_mean_max512"
DEFAULT_Y_GROUP = "all"
DEFAULT_SPLIT = "leave_games_out"
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_TEST_GAMES = 3
DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_VAL_SPLIT = "tail"
DEFAULT_VAL_GAMES = 1
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_OVERFIT_GAP_THRESHOLD = 0.0
DEFAULT_OVERFIT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.1
DEFAULT_HIDDEN_STR = "512,256"
DEFAULT_TOP_PERCENT = 35.0
DEFAULT_MUSIC_MODE = "min_loss"


def register_shared_rubert_mlp_cli(
    p: ArgumentParser,
    *,
    parse_hidden: Callable[[str], list[int]],
    device_default: str,
    include_embedding_lags: bool,
) -> None:
    p.add_argument("--dsn", default=None)
    p.add_argument("--rubert-table", default=TEXT_MUSICS_RUBERT_TABLE)
    p.add_argument(
        "--embedding-column",
        default=DEFAULT_EMBEDDING_COLUMN,
    )
    p.add_argument(
        "--merge-identical-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="слияние подряд идущих строк одной игры с одинаковым эмбеддингом и тем же music_id (по умолчанию вкл.; --no-merge-identical-embeddings — выкл.)",
    )
    p.add_argument("--music-table", default="music_data")
    p.add_argument(
        "--y-group",
        choices=("librosa", "jamendo", "all"),
        default=DEFAULT_Y_GROUP,
    )
    p.add_argument(
        "--split",
        choices=("within_game", "leave_games_out"),
        default=DEFAULT_SPLIT,
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )
    p.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--held-out-games-seed",
        type=int,
        default=None,
        help="leave_games_out: сид shuffle игр в test; по умолчанию случайный (печатается).",
    )
    p.add_argument(
        "--held-out-game-min-rows",
        type=int,
        default=None,
        help="leave_games_out: test только из игр с числом строк >= этого (вместе с --held-out-game-max-rows).",
    )
    p.add_argument(
        "--held-out-game-max-rows",
        type=int,
        default=None,
        help="leave_games_out: test только из игр с числом строк <= этого (вместе с --held-out-game-min-rows).",
    )
    if include_embedding_lags:
        p.add_argument(
            "--embedding-lags",
            type=int,
            default=1,
            help="сколько предыдущих эмбеддингов той же игры конкатенировать (0 = как обычный MLP)",
        )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="доля val: row/tail; при game не используется",
    )
    p.add_argument(
        "--val-split",
        choices=("row", "game", "tail"),
        default=DEFAULT_VAL_SPLIT,
    )
    p.add_argument("--val-games", type=int, default=DEFAULT_VAL_GAMES)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument(
        "--overfit-gap-threshold",
        type=float,
        default=DEFAULT_OVERFIT_GAP_THRESHOLD,
    )
    p.add_argument(
        "--overfit-patience",
        type=int,
        default=DEFAULT_OVERFIT_PATIENCE,
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument(
        "--hidden",
        type=parse_hidden,
        default=parse_hidden(DEFAULT_HIDDEN_STR),
        help="размеры скрытых слоёв через запятую, напр. 512,256",
    )
    p.add_argument("--standardize-x", action="store_true")
    p.add_argument("--no-standardize-y", action="store_true")
    p.add_argument("--device", default=device_default)
    p.add_argument("--top-percent", type=float, default=DEFAULT_TOP_PERCENT)
    p.add_argument("--per-column-metrics-csv", type=Path, default=None)
    p.add_argument("--skip-experiment-log", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-test-mae-norm-each-epoch", action="store_true")
    p.add_argument(
        "--music-mode",
        choices=("first", "second", "avg_loss", "min_loss"),
        default=DEFAULT_MUSIC_MODE,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="load x.npy/y.npy/meta from folder; no Postgres for data (use --skip-experiment-log in Colab).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write stdout.log, config.json, metrics.json; default per-column csv path in this dir.",
    )
    register_retrieval_column_mask_cli(p)


def register_autoreg_lags_only_cli(p: ArgumentParser) -> None:
    p.add_argument(
        "--pred-lags",
        type=int,
        default=2,
        help="сколько предыдущих выходов (норм. Y) конкатенировать в вход; начало игры — нули",
    )
    p.add_argument(
        "--streak-scale",
        type=float,
        default=32.0,
        help="делитель для float-признака streak (целое число подряд одинаковых nearest-треков по MAE в каталоге игры)",
    )
    p.add_argument(
        "--sample-warmup-epochs",
        type=int,
        default=5,
        help="эпохи с p=0 (в канал истории только истинный y_norm)",
    )
    p.add_argument(
        "--sample-p-final",
        type=float,
        default=0.5,
        help="целевая вероятность подставлять ŷ_norm в историю после разогрева",
    )
    p.add_argument(
        "--no-sample-linear-ramp",
        action="store_true",
        help="после разогрева сразу sample-p-final, без линейного роста от 0",
    )
