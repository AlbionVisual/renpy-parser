from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from text_scripts.ml_scripts.db_consts import (
    CREATE_TEXT_MUSICS_RUBERT_SQL,
    SOURCE_TABLE,
    TEXT_MUSICS_RUBERT_TABLE,
)

load_dotenv()


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Создаёт таблицу text_musics_rubert (если нет) и заполняет из text_musics_rel "
            "(game, phrase_order, text1, music). Исходная таблица не изменяется."
        ),
    )
    p.add_argument(
        "--truncate",
        action="store_true",
        help="TRUNCATE целевой таблицы перед вставкой",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help="Строка подключения psycopg; иначе env pghost",
    )
    args = p.parse_args()

    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен dsn или переменная окружения pghost")
        return 2

    insert_sql = f"""
        insert into {TEXT_MUSICS_RUBERT_TABLE} (game, phrase_order, text1, music)
        select distinct on (game, phrase_order)
            game, phrase_order, text1, music
        from {SOURCE_TABLE}
        order by game, phrase_order, file, line_number
        on conflict (game, phrase_order) do update set
            text1 = excluded.text1,
            music = excluded.music;
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TEXT_MUSICS_RUBERT_SQL)
            if args.truncate:
                cur.execute("truncate table " + TEXT_MUSICS_RUBERT_TABLE)
            cur.execute(insert_sql)
            cur.execute("select count(*) from " + TEXT_MUSICS_RUBERT_TABLE)
            nrows = cur.fetchone()[0]
        conn.commit()

    print("ok", TEXT_MUSICS_RUBERT_TABLE, "rows synced from", SOURCE_TABLE, "total rows", nrows)
    if args.truncate:
        print("truncate was True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
