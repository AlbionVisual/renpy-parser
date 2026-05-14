#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from consts import DOWNLOADS_DEFAULT, RAW_GAMES, WORK_DIR


def normPath(name):
    return name.replace("\\", "/")


def inGamePath(n):
    return "/game/" in n or n.startswith("game/")


def wantPath(name, no_file_filter=False):
    n = normPath(name)
    if name.endswith("/"):
        return False
    if no_file_filter:
        return True
    if n.endswith(".rpa"):
        return True
    if inGamePath(n):
        return True
    return False


def firstTopFolder(names):
    for n in names:
        n = normPath(n).strip()
        if not n or n.endswith("/"):
            continue
        parts = n.split("/")
        if not parts:
            continue
        candidate = parts[0].strip()
        if not candidate:
            continue
        if candidate.replace(".", "").replace("-", "").replace(" ", "").isdigit():
            continue
        if len(candidate) == 8 and candidate.isdigit():
            continue
        return candidate
    return None


def archiveHasGame(names):
    for n in names:
        if inGamePath(normPath(n)) or normPath(n).endswith(".rpa"):
            return True
    return False


def peekZip(archive_path):
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            names = z.namelist()
    except Exception as e:
        print("  ошибка открытия zip:", e)
        return None, False, []
    first = firstTopFolder(names)
    has_game = archiveHasGame(names)
    return first, has_game, names


def line7zPath(line):
    tokens = line.split()
    for i, t in enumerate(tokens):
        if "/" in t or "\\" in t:
            return " ".join(tokens[i:]).replace("\\", "/")
    return line.replace("\\", "/")


def peek7z(archive_path):
    try:
        res = subprocess.run(
            ["7z", "l", "-ba", str(archive_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print("  не найден 7z")
        return None, False, []
    if res.returncode != 0:
        return None, False, []
    raw = [s.strip() for s in res.stdout.strip().split("\n") if s.strip()]
    lines = [line7zPath(s) for s in raw]
    first = firstTopFolder(lines)
    has_game = archiveHasGame(lines)
    return first, has_game, lines


def peekRar(archive_path):
    try:
        res = subprocess.run(
            ["unrar", "lb", str(archive_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print("  не найден unrar")
        return None, False, []
    if res.returncode != 0:
        return None, False, []
    lines = [s.strip().replace("\\", "/")
             for s in res.stdout.strip().split("\n") if s.strip()]
    first = firstTopFolder(lines)
    has_game = archiveHasGame(lines)
    return first, has_game, lines


def pullZip(archive_path, target_dir, to_extract):
    print("  извлекаю zip, файлов:", len(to_extract))
    with zipfile.ZipFile(archive_path, "r") as z:
        for n in to_extract:
            z.extract(n, target_dir)
    for n in to_extract[:10]:
        print("    ", n)
    if len(to_extract) > 10:
        print("    ... и ещё", len(to_extract) - 10)


def pull7z(archive_path, target_dir, to_extract):
    print("  извлекаю 7z, файлов:", len(to_extract))
    for n in to_extract:
        subprocess.run(
            ["7z", "x", "-y", "-o" + str(target_dir), str(archive_path), n],
            capture_output=True,
            timeout=30,
        )
    for n in to_extract[:10]:
        print("    ", n)
    if len(to_extract) > 10:
        print("    ... и ещё", len(to_extract) - 10)


def pullRar(archive_path, target_dir, to_extract):
    print("  извлекаю rar, файлов:", len(to_extract))
    for n in to_extract:
        subprocess.run(
            ["unrar", "x", "-o+", "-inul",
                str(archive_path), n, str(target_dir) + "/"],
            capture_output=True,
            timeout=30,
        )
    for n in to_extract[:10]:
        print("    ", n)
    if len(to_extract) > 10:
        print("    ... и ещё", len(to_extract) - 10)


def runArchives(args):
    downloads = Path(args.downloads)
    print("папка загрузок:", downloads)
    if not downloads.is_dir():
        print("папка не найдена")
        return 1, {}

    exts = ("*.zip", "*.7z", "*.rar")
    archives = []
    for ext in exts:
        archives.extend(downloads.glob(ext))
    archives = [p for p in archives if p.is_file()]

    if not archives:
        print("архивов не найдено")
        return 0, {}

    print("найдено архивов:", len(archives))
    RAW_GAMES.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        print("рабочая папка в WSL:", WORK_DIR)

    archive_mapping = {}

    for archive_path in archives:
        print("\nархив:", archive_path.name)
        suf = archive_path.suffix.lower()

        if suf == ".zip":
            first_folder, has_game, names = peekZip(archive_path)
        elif suf == ".7z":
            first_folder, has_game, names = peek7z(archive_path)
        elif suf == ".rar":
            first_folder, has_game, names = peekRar(archive_path)
        else:
            continue

        if not names:
            print("\033[91m  не удалось прочитать архив, пропуск\033[0m")
            continue

        if first_folder is None:
            print("\033[91m  не удалось определить первую папку, пропуск\033[0m")
            continue

        if (RAW_GAMES / first_folder).exists():
            print("\033[91m  папка уже есть в raw-games:", first_folder, ", пропуск\033[0m")
            continue

        if not has_game:
            print("\033[91m  в архиве нет папки game, пропуск\033[0m")
            continue

        to_extract = [n for n in names if wantPath(n, args.no_file_filter)]
        if not to_extract:
            print("\033[91m  нет извлекаемых путей (ожидается game/** или *.rpa), пропуск\033[0m")
            continue

        if args.no_copy:
            work_path = archive_path
        else:
            work_path = WORK_DIR / archive_path.name
            print("  копирую в WSL:", work_path)
            try:
                shutil.copy2(archive_path, work_path)
            except Exception as e:
                print("  ошибка копирования:", e)
                continue

        target = RAW_GAMES / first_folder
        print("  цель:", target)

        if suf == ".zip":
            pullZip(work_path, target, to_extract)
        elif suf == ".7z":
            pull7z(work_path, target, to_extract)
        elif suf == ".rar":
            pullRar(work_path, target, to_extract)

        game_dir = target / "game"
        if game_dir.is_dir():
            print("  перемещаю содержимое game в корень игры")
            for item in game_dir.iterdir():
                raw_games_item = target / item.name
                if raw_games_item.exists():
                    if raw_games_item.is_dir():
                        shutil.rmtree(raw_games_item)
                    else:
                        raw_games_item.unlink()
                shutil.move(str(item), str(target))
            shutil.rmtree(game_dir)

        if not args.no_copy and work_path.exists():
            work_path.unlink()
            print("  временный файл удалён")

        archive_mapping[first_folder] = archive_path

    print("\nготово. Результат в", RAW_GAMES)
    return 0, archive_mapping


if __name__ == "__main__":
    pr = argparse.ArgumentParser()
    pr.add_argument("--no-copy", action="store_true")
    pr.add_argument("--downloads", default=DOWNLOADS_DEFAULT)
    pr.add_argument("--no-file-filter", action="store_true")
    a = pr.parse_args()
    code, _ = runArchives(a)
    sys.exit(code)
