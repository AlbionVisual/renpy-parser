from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from consts import TABLE_NAME

load_dotenv()


def ensure_text1_column(cur) -> None:
    cur.execute(
        sql.SQL("alter table {} add column if not exists text1 text").format(
            sql.Identifier(TABLE_NAME),
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Заполняет text1 в text_musics_rel из text_orig / text_translated "
        "без input(): приоритет перевода или оригинала.",
    )
    p.add_argument(
        "--prefer",
        choices=("translated", "orig", "smart"),
        default="smart",
        help="translated|orig: жёсткий приоритет колонки; smart: если в orig есть "
        "кириллица, а в translated только латиница — берём orig (иначе как translated)",
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="пересчитать text1 для всех строк с непустым orig/trans, не только "
        "где text1 is null (нужно, чтобы исправить уже залитый английский)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="только посчитать затронутые строки, без UPDATE",
    )
    p.add_argument("--dsn", default=None, help="psycopg dsn; иначе env pghost")
    args = p.parse_args()

    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен --dsn или переменная окружения pghost")
        return 2

    ru_in_orig_en_in_trans = """
        text_orig is not null and btrim(text_orig) <> ''
        and text_translated is not null and btrim(text_translated) <> ''
        and text_orig ~ '[А-Яа-яЁё]'
        and text_translated !~ '[А-Яа-яЁё]'
        and text_translated ~ '[A-Za-z]'
    """

    if args.prefer == "translated":
        value_sql = """
            case
              when text_translated is not null and btrim(text_translated) <> ''
                then btrim(text_translated)
              when text_orig is not null and btrim(text_orig) <> ''
                then btrim(text_orig)
              else text1
            end
        """
    elif args.prefer == "orig":
        value_sql = """
            case
              when text_orig is not null and btrim(text_orig) <> ''
                then btrim(text_orig)
              when text_translated is not null and btrim(text_translated) <> ''
                then btrim(text_translated)
              else text1
            end
        """
    else:
        value_sql = """
            case
              when """ + ru_in_orig_en_in_trans + """
                then btrim(text_orig)
              when text_translated is not null and btrim(text_translated) <> ''
                then btrim(text_translated)
              when text_orig is not null and btrim(text_orig) <> ''
                then btrim(text_orig)
              else text1
            end
        """

    has_src = """
        (text_translated is not null and btrim(text_translated) <> '')
        or (text_orig is not null and btrim(text_orig) <> '')
    """
    if args.all_rows:
        where_sql = "(" + has_src + ")"
    else:
        where_sql = "text1 is null and (" + has_src + ")"

    count_q = f"select count(*) from {TABLE_NAME} where " + where_sql
    update_q = f"""
        update {TABLE_NAME}
        set text1 = {value_sql}
        where {where_sql}
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            ensure_text1_column(cur)
            cur.execute(count_q)
            n = cur.fetchone()[0]
            if args.all_rows:
                print("rows to update (has orig/trans):", n)
            else:
                print("rows to fill (text1 is null, has orig/trans):", n)
            if args.dry_run:
                conn.rollback()
                return 0
            cur.execute(update_q)
            print("updated rows:", cur.rowcount)
        conn.commit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
