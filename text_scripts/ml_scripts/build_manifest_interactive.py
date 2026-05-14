from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prompt(label: str, default: str | None = None) -> str:
    if default is None:
        s = input(label + " ").strip()
        return s
    s = input(label + " [" + default + "]: ").strip()
    return s if s else default


def _prompt_int(label: str, default: int) -> int:
    s = _prompt(label, str(default))
    return int(s)


def _prompt_yes(label: str, default_yes: bool) -> bool:
    d = "y" if default_yes else "n"
    s = _prompt(label + " (y/n)", d).lower()
    return s in ("y", "yes", "1", "да")


def _model_menu() -> dict[str, tuple[str, str, list[str]]]:
    return {
        "1": (
            "mlp",
            "text_scripts/ml_scripts/train_rubert_to_music_mlp.py",
            [],
        ),
        "2": (
            "mlp_lags",
            "text_scripts/ml_scripts/train_rubert_to_music_mlp_lags.py",
            [],
        ),
        "3": (
            "gru",
            "text_scripts/ml_scripts/train_rubert_to_music_gru.py",
            [],
        ),
        "4": (
            "lstm",
            "text_scripts/ml_scripts/train_rubert_to_music_lstm.py",
            [],
        ),
        "5": (
            "tcn",
            "text_scripts/ml_scripts/train_rubert_to_music_tcn.py",
            [],
        ),
        "6": (
            "transformer_window",
            "text_scripts/ml_scripts/train_rubert_to_music_transformer_window.py",
            [],
        ),
        "7": (
            "mlp_autoreg_lags",
            "text_scripts/ml_scripts/train_rubert_to_music_mlp_autoreg_lags.py",
            [],
        ),
    }


def _extra_args_for_model(key: str, shared_lags: int) -> list[str]:
    out: list[str] = []
    if key in ("mlp_lags", "gru", "lstm", "tcn", "transformer_window", "mlp_autoreg_lags"):
        out.extend(["--embedding-lags", str(shared_lags)])

    if key == "mlp":
        h = _prompt("  MLP --hidden (через запятую)", "512,256")
        out.extend(["--hidden", h])

    if key == "mlp_lags":
        h = _prompt("  MLP+lags --hidden (через запятую)", "512,256")
        out.extend(["--hidden", h])

    if key == "gru":
        out.extend(
            [
                "--gru-hidden",
                str(_prompt_int("  GRU --gru-hidden", 256)),
                "--gru-layers",
                str(_prompt_int("  GRU --gru-layers", 1)),
            ]
        )

    if key == "lstm":
        out.extend(
            [
                "--lstm-hidden",
                str(_prompt_int("  LSTM --lstm-hidden", 256)),
                "--lstm-layers",
                str(_prompt_int("  LSTM --lstm-layers", 1)),
            ]
        )

    if key == "tcn":
        out.extend(
            [
                "--tcn-channels",
                str(_prompt_int("  TCN --tcn-channels", 256)),
                "--tcn-levels",
                str(_prompt_int("  TCN --tcn-levels", 6)),
                "--tcn-kernel",
                str(_prompt_int("  TCN --tcn-kernel", 3)),
            ]
        )

    if key == "transformer_window":
        out.extend(
            [
                "--tf-window",
                str(_prompt_int("  Transformer --tf-window", 512)),
                "--tf-d-model",
                str(_prompt_int("  Transformer --tf-d-model", 256)),
                "--tf-layers",
                str(_prompt_int("  Transformer --tf-layers", 4)),
                "--tf-heads",
                str(_prompt_int("  Transformer --tf-heads", 4)),
                "--tf-dim-ff",
                str(_prompt_int("  Transformer --tf-dim-ff", 1024)),
            ]
        )

    if key == "mlp_autoreg_lags":
        out.extend(
            [
                "--hidden",
                _prompt("  autoreg --hidden", "512,256"),
                "--pred-lags",
                str(_prompt_int("  autoreg --pred-lags", 2)),
            ]
        )

    return out


def _build_shared_argv(
    *,
    device: str,
    epochs: int,
    batch_size: int,
    split: str,
    test_fraction: float,
    test_games: int,
    val_split: str,
    val_fraction: float,
    val_games: int,
    no_progress: bool,
) -> list[str]:
    argv = [
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--split",
        split,
        "--val-split",
        val_split,
        "--val-fraction",
        str(val_fraction),
    ]
    if split == "leave_games_out":
        argv.extend(["--test-games", str(test_games)])
    else:
        argv.extend(["--test-fraction", str(test_fraction)])
    if val_split == "game":
        argv.extend(["--val-games", str(val_games)])
    if no_progress:
        argv.append("--no-progress")
    return argv


def main() -> int:
    p = argparse.ArgumentParser(
        description="Интерактивно собрать manifest.jsonl для run_manifest.py (Colab / локально).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="куда записать manifest (по умолчанию спросит)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="дописать в конец файла вместо перезаписи",
    )
    args_ns = p.parse_args()

    print("")
    print("Сборщик manifest.jsonl для text_scripts/ml_scripts/run_manifest.py")
    print("Запускай из корня репозитория renpy-parser (пути script относительно корня).")
    print("Порядок заданий = порядок строк (оркестратор берёт сверху первый pending/failed).")
    print("")

    if args_ns.output is None:
        out_path = Path(_prompt("Путь к manifest.jsonl", "manifest.jsonl")).expanduser().resolve()
    else:
        out_path = args_ns.output.expanduser().resolve()

    device = _prompt("Устройство (--device)", "cuda")
    epochs = _prompt_int("Эпохи (--epochs)", 100)
    batch_size = _prompt_int("Batch (--batch-size)", 256)
    seed = _prompt_int("Сид обучения (пойдёт в JSON и в --seed)", 42)

    split = _prompt("Сплит: within_game | leave_games_out", "leave_games_out")
    if split not in ("within_game", "leave_games_out"):
        print("fatal: split должен быть within_game или leave_games_out")
        return 2

    test_fraction = float(_prompt("Если within_game: --test-fraction", "0.2"))
    test_games = _prompt_int("Если leave_games_out: --test-games", 3)
    held_seed = _prompt_int("--held-out-games-seed (для leave_games_out)", 42)

    val_split = _prompt("Val: row | tail | game", "tail")
    if val_split not in ("row", "tail", "game"):
        print("fatal: val-split должен быть row, tail или game")
        return 2
    val_fraction = float(_prompt("--val-fraction (для row/tail)", "0.1"))
    val_games = _prompt_int("--val-games (для val-split game)", 1)

    shared_lags = _prompt_int(
        "Общий --embedding-lags для моделей с лагами (GRU/LSTM/…/lags)", 1
    )

    no_progress = _prompt_yes("--no-progress (уменьшит шум в логе)", True)

    shared_argv = _build_shared_argv(
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        split=split,
        test_fraction=test_fraction,
        test_games=test_games,
        val_split=val_split,
        val_fraction=val_fraction,
        val_games=val_games,
        no_progress=no_progress,
    )

    menu = _model_menu()
    rows: list[dict[str, object]] = []

    print("")
    print("Модели:")
    for k, (name, _path, _) in menu.items():
        print(" ", k, name)
    print(" 0 — закончить добавление заданий")
    print("")

    idx = 0
    while True:
        choice = _prompt("Номер модели (0 = готово)", "0").strip()
        if choice == "0":
            break
        if choice not in menu:
            print("Неизвестный номер, попробуй снова.")
            continue
        tag, script, _ = menu[choice]
        idx += 1
        default_rid = "run_" + str(idx).zfill(3) + "_" + tag
        run_id = _prompt("  run_id (папка run на Drive)", default_rid)

        base_argv = list(shared_argv)
        base_argv.extend(_extra_args_for_model(tag, shared_lags))

        row: dict[str, object] = {
            "run_id": run_id,
            "script": script,
            "args": base_argv,
            "seed": seed,
            "held_out_games_seed": held_seed,
            "status": "pending",
            "output_dir": "",
        }
        rows.append(row)
        print("  добавлено:", run_id, script)
        print("")

    if not rows:
        print("Нет строк — выход без записи.")
        return 0

    mode = "a" if args_ns.append and out_path.exists() else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print("")
    print("Записано строк:", len(rows), "→", str(out_path))
    print("Дальше на Colab положи этот файл на Drive и укажи тот же путь в MANIFEST_PATH.")
    print("Локально: python text_scripts/ml_scripts/run_manifest.py --manifest ... --base-run-dir ... --dataset-dir ... --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
