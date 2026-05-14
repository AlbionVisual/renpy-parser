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

COUNT_ORPHANS = """
select count(*) from music_data m
where not exists (
  select 1 from {tbl} t where t.music = m.id
)
""".format(
    tbl=TABLE_NAME,
)

SAMPLE_ORPHANS = """
select m.id, m.game, m.path
from music_data m
where not exists (
  select 1 from {tbl} t where t.music = m.id
)
order by m.game, m.path
limit %s
""".format(
    tbl=TABLE_NAME,
)

DELETE_ORPHANS = """
delete from music_data m
where not exists (
  select 1 from {tbl} t where t.music = m.id
)
""".format(
    tbl=TABLE_NAME,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Удаляет из music_data строки, на которые ни одна запись "
        f"{TABLE_NAME} не ссылается через music (сироты). "
        "Такие треки ни к одной реплике не привязаны — их не нужно тащить в анализ по FK. "
        "Если в music_data уже писались DSP/прочие фичи (AudioEnricher и т.п.), "
        "у сирот они тоже пропадут вместе со строкой.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="выполнить DELETE; без флага только счётчик и пример строк",
    )
    p.add_argument("--sample", type=int, default=15, help="сколько примеров id/game/path показать")
    p.add_argument("--dsn", default=None, help="строка подключения; иначе env pghost")
    args = p.parse_args()
    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен --dsn или переменная окружения pghost")
        return 2

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(COUNT_ORPHANS)
            n = cur.fetchone()[0]
            print("orphan music_data rows:", n)
            if args.sample > 0 and n:
                cur.execute(SAMPLE_ORPHANS, (args.sample,))
                rows = cur.fetchall()
                print("sample (id, game, path):")
                for r in rows:
                    print(" ", r[0], "|", r[1], "|", r[2])
            if not args.apply:
                print("preview only. Pass --apply to delete orphans.")
                return 0
            cur.execute(DELETE_ORPHANS)
            deleted = cur.rowcount
        conn.commit()
        print("deleted:", deleted)
        with conn.cursor() as cur:
            cur.execute(COUNT_ORPHANS)
            n_after = cur.fetchone()[0]
        print("orphans remaining:", n_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
