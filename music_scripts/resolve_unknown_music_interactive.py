from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
import psycopg
from psycopg import sql

from consts import AUDIOS_DIR, TABLE_NAME

load_dotenv()

_AUDIO_EXT = frozenset({".ogg", ".opus", ".mp3", ".wav"})


def _norm_rel(p: str) -> str:
    return (p or "").strip().replace("\\", "/").lstrip("/")


def list_game_audio_rels(game: str) -> list[str]:
    root = AUDIOS_DIR / game
    if not root.is_dir():
        return []
    out: list[str] = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in _AUDIO_EXT:
            out.append(fp.relative_to(root).as_posix())
    return sorted(out)


def _load_sync_columns():
    p = _REPO_ROOT / "music_scripts" / "parse_folder_to_db.py"
    spec = importlib.util.spec_from_file_location("parse_folder_to_db", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.sync_columns


def fetch_pairs(
    cur,
    *,
    table: str,
    only_unknown_prefix: bool,
    game_filter: str | None,
) -> list[tuple[str, str, int, str | None]]:
    parts: list[sql.SQL] = [
        sql.SQL("music is null"),
        sql.SQL("coalesce(trim(music_path), '') <> 'NO_MUSIC'"),
    ]
    if only_unknown_prefix:
        parts.append(sql.SQL("trim(music_path) ilike 'unknown:%'"))
    if game_filter:
        parts.append(sql.SQL("game = %s"))
    cond = sql.SQL(" and ").join(parts)
    q = sql.SQL(
        """
        select game, music_path, count(*)::int as n,
               max(raw_music_line) as sample_line
        from {tbl}
        where {cond}
        group by game, music_path
        order by game, music_path
        """
    ).format(tbl=sql.Identifier(table), cond=cond)
    params: list[object] = [game_filter] if game_filter else []
    cur.execute(q, params)
    return [(str(a), str(b), int(c), d) for a, b, c, d in cur.fetchall()]


def music_data_has_path(cur, game: str, path: str) -> bool:
    cur.execute(
        "select 1 from music_data where game = %s and path = %s limit 1",
        (game, path),
    )
    return cur.fetchone() is not None


def ensure_music_data_row(cur, game: str, path: str) -> None:
    cur.execute(
        """
        insert into music_data (game, path)
        select %s, %s
        where not exists (select 1 from music_data m where m.game = %s and m.path = %s)
        """,
        (game, path, game, path),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="По очереди: пары (game, music_path) без music — вводите канонический "
        "относительный путь к файлу под audios/<game>/; при совпадении с диском или "
        "music_data обновляется music_path и вызывается sync_columns().",
    )
    p.add_argument("--dsn", default=None, help="psycopg DSN; иначе env pghost")
    p.add_argument(
        "--only-unknown-prefix",
        action="store_true",
        help="только строки, где music_path начинается с UNKNOWN: (без учёта регистра)",
    )
    p.add_argument("--game", default=None, help="ограничить одной игрой")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="не писать в БД, только показать что бы сделали",
    )
    p.add_argument(
        "--max-hints",
        type=int,
        default=12,
        help="сколько путей из audios/<game> показать как подсказку (0 = не показывать)",
    )
    args = p.parse_args()
    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен --dsn или переменная окружения pghost")
        return 2

    sync_columns = _load_sync_columns()
    table = TABLE_NAME

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            pairs = fetch_pairs(
                cur,
                table=table,
                only_unknown_prefix=bool(args.only_unknown_prefix),
                game_filter=args.game,
            )
    if not pairs:
        print("Нет пар (game, music_path) по условию.")
        return 0

    print("Всего уникальных пар:", len(pairs))
    print("Команды ввода: пустая строка = пропуск, q = выход")
    print("Путь: относительно папки игры, как в music_data.path (напр. audio/foo.ogg)")

    for game, old_path, n_rows, sample in pairs:
        print("")
        print("---")
        print("game:", game)
        print("music_path:", repr(old_path))
        print("rows:", n_rows)
        if sample:
            print("sample raw_music_line:", repr(sample)[:500])
        if args.max_hints > 0:
            rels = list_game_audio_rels(game)
            if rels:
                print("hints (first", args.max_hints, "of", len(rels), "audio files):")
                for h in rels[: args.max_hints]:
                    print(" ", h)
            else:
                print("hints: (no files under", str(AUDIOS_DIR / game), ")")

        try:
            raw = input("new path> ").strip()
        except EOFError:
            print("EOF, exit")
            break
        if raw.lower() == "q":
            print("quit")
            break
        if not raw:
            print("skip")
            continue

        new_path = _norm_rel(raw)
        full = AUDIOS_DIR / game / new_path
        on_disk = full.is_file()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                in_catalog = music_data_has_path(cur, game, new_path)
                if not on_disk and not in_catalog:
                    print("нет файла", str(full), "и нет строки music_data(game, path)")
                    continue
                if args.dry_run:
                    print("dry-run: would set music_path ->", repr(new_path), "for", n_rows, "rows")
                    continue
                ensure_music_data_row(cur, game, new_path)
                cur.execute(
                    sql.SQL(
                        "update {tbl} set music_path = %s "
                        "where game = %s and music_path = %s and music is null"
                    ).format(tbl=sql.Identifier(table)),
                    (new_path, game, old_path),
                )
                n = cur.rowcount
            conn.commit()
        print("updated rows:", n, "music_path ->", repr(new_path))
        sync_columns()
        print("sync_columns() done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
