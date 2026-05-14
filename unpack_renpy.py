#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path
import sys
from consts import UNRPYC_PATH, RAW_GAMES, AUDIOS_DIR


def checkTools():
    unrpa = shutil.which("unrpa")
    if unrpa is None:
        local_bin = Path.home() / ".local" / "bin" / "unrpa"
        if local_bin.exists():
            unrpa = str(local_bin)

    if not UNRPYC_PATH.exists():
        print("\n" + "!" * 60)
        print(" ОШИБКА: Не найден декомпилятор unrpyc.")
        print(" Ожидаемый путь:", UNRPYC_PATH)
        print(" Попробуйте выполнить: git clone https://github.com/CensoredUsername/unrpyc.git tools/unrpyc")
        print("!" * 60 + "\n")
        sys.exit(1)

    if unrpa is None:
        print("\n" + "!" * 60)
        print(" ОШИБКА: Не найден распаковщик unrpa.")
        print(" Пожалуйста, установите его: pip install unrpa --break-system-packages")
        print("!" * 60 + "\n")
        sys.exit(1)

    return unrpa


def runCmd(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        return True
    except Exception as e:
        print("    [!] Ошибка запуска", cmd[0], ":", e)
        return False


def moveAudio(game_dir, game_name):
    target_dir = AUDIOS_DIR / game_name
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = frozenset({".ogg", ".opus", ".mp3", ".wav"})

    for f in game_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in audio_extensions:
            rel_path = f.relative_to(game_dir)
            dest_path = target_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest_path))


def processFolder(game_path, unrpa_cmd, no_file_filter=False):
    print("\n>>> ИГРА:", game_path.name)

    game_dirs = list(game_path.rglob("game"))
    if game_dirs:
        game_dir = game_dirs[0]
    else:
        game_dir = game_path

    rpy_before = len(list(game_dir.glob("*.rpy")))

    rpa_files = list(game_dir.rglob("*.rpa"))
    for rpa in rpa_files:
        print("    [+] Распаковка", rpa.name, "...")
        if runCmd([unrpa_cmd, "-p", str(game_dir), str(rpa)], cwd=game_dir):
            rpa.unlink()

    print("    [V] Распаковка RPA завершена (ассеты в game/ сохраняются целиком).")

    rpyc_files = list(game_dir.rglob("*.rpyc"))
    for rpyc in rpyc_files:
        rpy = rpyc.with_suffix(".rpy")
        if not rpy.exists():
            print("    [+] Декомпиляция", rpyc.name, "...")
            runCmd([sys.executable, str(UNRPYC_PATH), str(rpyc)], cwd=game_dir)

    rpy_after = list(game_dir.glob("*.rpy"))

    if len(rpy_after) == 0 and rpy_before == 0:
        print(
            "    [!] ВНИМАНИЕ: в game/ нет ни одного .rpy (и unrpyc ничего не добавил), пропуск.")
        return

    print("    [V] Декомпиляция rpyc завершена.")

    print("    [*] Перемещение аудиофайлов в централизованную папку...")
    moveAudio(game_dir, game_path.name)

    print("    [V] Готово.")


def unpackRenpy(no_file_filter=False):
    unrpa_cmd = checkTools()
    if not RAW_GAMES.exists():
        print("Папка", RAW_GAMES, "не существует.")
        return
    for folder in [f for f in RAW_GAMES.iterdir() if f.is_dir()]:
        processFolder(folder, unrpa_cmd, no_file_filter)


if __name__ == "__main__":
    unpackRenpy("--no-file-filter" in sys.argv)
