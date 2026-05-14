from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
import psycopg

from consts import TABLE_NAME

load_dotenv()

CONFUSED_SQL = """
select game, music_path, sfx_before_path, raw_music_line, music
from {table}
where music is null
  and coalesce(trim(music_path), '') <> 'NO_MUSIC'
order by game, music_path, file, line_number
limit %s
""".format(
    table=TABLE_NAME,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Печатает первые N «плохих» строк (music is null, music_path не NO_MUSIC), "
        "аналог get_confused_cols без input.",
    )
    p.add_argument("--limit", type=int, default=10, help="сколько строк показать")
    p.add_argument(
        "--dsn",
        default=None,
        help="строка подключения psycopg; иначе env pghost",
    )
    args = p.parse_args()
    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен --dsn или переменная окружения pghost")
        return 2
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CONFUSED_SQL, (args.limit,))
            rows = cur.fetchall()
    if not rows:
        print("Нет строк по условию (всё связано или только NO_MUSIC).")
        return 0
    print("rows:", len(rows))
    print("columns: game | music_path | sfx_before_path | raw_music_line | music")
    for r in rows:
        print("---")
        for i, col in enumerate(
            ("game", "music_path", "sfx_before_path", "raw_music_line", "music")
        ):
            print(col, ":", r[i])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
