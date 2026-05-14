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

load_dotenv()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Выполняет SQL в PostgreSQL (pghost или --dsn). "
        "Для SELECT печатает строки; для DDL/DML — rowcount.",
    )
    p.add_argument(
        "-c",
        "--command",
        action="append",
        dest="commands",
        default=[],
        help="SQL-команда (можно несколько раз)",
    )
    p.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        help="файл с SQL (выполняется целиком одним execute)",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help="строка подключения; иначе env pghost",
    )
    args = p.parse_args()
    dsn = args.dsn or os.environ.get("pghost")
    if not dsn:
        print("Нужен --dsn или переменная окружения pghost")
        return 2
    parts = list(args.commands or [])
    if args.file:
        parts.append(args.file.read_text(encoding="utf-8"))
    if not parts:
        print("Укажите -c \"SQL\" или -f path.sql")
        return 2
    sql_blob = "\n".join(parts)
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql_blob)
            if cur.description:
                colnames = [d.name for d in cur.description]
                print("\t".join(colnames))
                for row in cur.fetchall():
                    print("\t".join("" if v is None else str(v) for v in row))
            else:
                print("ok rowcount", cur.rowcount)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
