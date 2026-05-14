#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

from consts import RAW_GAMES, OUTPUT_DIR, FIELDNAMES
from scan_rpy_dir import scanGame
from trace_labels import traceGame, saveCsv
from csv_sync_db import insertDataset


def runOneGame(game_dir):
    print("\n=== Игра:", game_dir.name, "===")

    scan_data = scanGame(game_dir)
    if scan_data is None:
        print("  нет .rpy файлов, пропуск")
        return None

    print("  файлов:", len(scan_data["rpy_files"]))
    print("  labels:", len(scan_data["labels"]))
    print("  audio defines:", len(scan_data["audio_map"]))
    print("  music dict keys:", len(scan_data.get("music_subscript_paths", {})))

    if "start" not in scan_data["labels"]:
        all_labels = list(scan_data["labels"].keys())
        if not all_labels:
            print("  WARNING: меток не найдено, пропуск")
            return None

        start_label = all_labels[0]
        print(
            "  WARNING: label 'start' не найден, начинаем с '" + start_label + "'")
    else:
        start_label = "start"

    dataset, warnings = traceGame(scan_data, start_label=start_label)

    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print("    -", w)

    if not dataset:
        print("  датасет пуст (нет диалогов)")
        return None

    print("  строк диалогов:", len(dataset))

    translated_count = sum(1 for r in dataset if r.get(
        FIELDNAMES["text from translations"]))
    if translated_count:
        print("  переведено:", translated_count, "/", len(dataset))
    else:
        print("  перевод: не найден")

    music_tracks = set(r[FIELDNAMES["path to bg music"]]
                       for r in dataset if r[FIELDNAMES["path to bg music"]] != "NO_MUSIC")
    print("  уникальных треков:", len(music_tracks))
    for t in sorted(music_tracks):
        print("    ", t)

    csv_path = OUTPUT_DIR / (game_dir.name + ".csv")
    saveCsv(dataset, csv_path)

    return dataset, csv_path


def runParse(args, archive_mapping=None):
    if archive_mapping is None:
        archive_mapping = {}

    if not RAW_GAMES.exists():
        print("Папка raw-games не найдена:", RAW_GAMES)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_dirs = sorted([d for d in RAW_GAMES.iterdir() if d.is_dir()])
    if not game_dirs:
        print("В raw-games нет папок с играми")
        return 1

    print("Найдено игр:", len(game_dirs))

    processed_games = []
    for game_dir in game_dirs:
        res = runOneGame(game_dir)
        if res:
            dataset, csv_path = res
            processed_games.append((game_dir, dataset, csv_path))

    print("\n=== Этап сохранения в БД ===")
    for game_dir, dataset, csv_path in processed_games:
        ans = input(
            "\nИгра " + game_dir.name + ": записать в БД и удалить временные файлы [Y/n]? ").strip().lower()
        if ans == "" or ans == "y":
            try:
                insertDataset(dataset)
                success = True
            except Exception as e:
                print("  ошибка при записи в БД:", e)
                success = False

            if success:
                archive_path = archive_mapping.get(game_dir.name)
                if archive_path and archive_path.exists():
                    try:
                        archive_path.unlink()
                        print("  исходный архив удалён:", archive_path)
                    except Exception as e:
                        print("  не удалось удалить архив:", e)

                try:
                    shutil.rmtree(game_dir)
                    print("  папка игры удалена:", game_dir)
                except Exception as e:
                    print("  не удалось удалить папку игры:", e)

                try:
                    csv_path.unlink()
                    print("  CSV файл удалён:", csv_path)
                except Exception as e:
                    print("  не удалось удалить CSV файл:", e)
        else:
            print("  сохранение в БД и удаление пропущено.")

    print("\nГотово. Результаты в", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    pr = argparse.ArgumentParser()
    a = pr.parse_args()
    sys.exit(runParse(a, {}))
