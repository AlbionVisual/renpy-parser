import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from consts import TABLE_NAME

load_dotenv()


def ensure_text1_column(cur, conn):
    cur.execute(
        sql.SQL("alter table {} add column if not exists text1 text").format(
            sql.Identifier(TABLE_NAME)
        )
    )
    conn.commit()


def games_needing_text1(cur):
    cur.execute(
        """
        select game
        from text_musics_rel
        group by game
        having bool_and(text1 is null)
           and bool_or(text_orig is not null or text_translated is not null)
        order by game
        """
    )
    return [r[0] for r in cur.fetchall()]


def sample_rows(cur, game, limit=10):
    cur.execute(
        """
        select text_orig, text_translated
        from text_musics_rel
        where game = %s
          and (text_orig is not null or text_translated is not null)
        order by file, line_number
        limit %s
        """,
        (game, limit),
    )
    return cur.fetchall()


def truncate(s, max_len=280):
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def ask_column(game, rows):
    print("\nИгра:", game)
    print("Примеры (text_orig | text_translated), не более 10 строк:\n")
    for i, (orig, trans) in enumerate(rows, 1):
        print("---", i, "---")
        print("text_orig:        ", truncate(orig))
        print("text_translated:  ", truncate(trans))
        print()
    while True:
        print(
            "Колонку записать в text1 для всей игры: [o] text_orig, "
            "[t] text_translated, [s] пропустить игру, [q] выход"
        )
        choice = input().strip().lower()
        if choice in ("o", "orig", "original", "1", "ор", "о"):
            return "text_orig"
        if choice in ("t", "trans", "translated", "2", "тр", "т"):
            return "text_translated"
        if choice in ("s", "skip", ""):
            return None
        if choice in ("q", "quit", "exit"):
            return "QUIT"
        print("Не понял ввод, повторите.")


def apply_update(cur, conn, game, column):
    cur.execute(
        sql.SQL("update {} set text1 = {} where game = %s").format(
            sql.Identifier(TABLE_NAME),
            sql.Identifier(column),
        ),
        (game,),
    )
    n = cur.rowcount
    conn.commit()
    print("Обновлено строк:", n, "игра:", game, "поле:", column)


def main():
    conn_str = os.environ.get("pghost")
    if not conn_str:
        print("Ошибка: в окружении нет переменной pghost")
        sys.exit(1)

    conn = psycopg.connect(conn_str)
    cur = conn.cursor()

    ensure_text1_column(cur, conn)

    games = games_needing_text1(cur)
    if not games:
        print(
            "Нет игр без text1 (при этом есть хотя бы одна строка с "
            "text_orig/text_translated)."
        )
        conn.close()
        return

    print("Игр к обработке:", len(games))

    for game in games:
        rows = sample_rows(cur, game)
        if not rows:
            print("Пропуск (нет строк с текстом):", game)
            continue
        col = ask_column(game, rows)
        if col == "QUIT":
            print("Выход без дальнейших обновлений.")
            break
        if col is None:
            print("Игра пропущена:", game)
            continue
        apply_update(cur, conn, game, col)

    conn.close()


if __name__ == "__main__":
    main()
