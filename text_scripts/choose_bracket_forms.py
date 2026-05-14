import os
import re
import sys
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from consts import TABLE_NAME

load_dotenv()


BRACKET_FORM_RE = re.compile(r"\[([A-Za-zА-Яа-яЁё]+)_([A-Za-zА-Яа-яЁё]+)\]")

LAT_TO_CYR = str.maketrans(
    {
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
        "k": "к",
        "m": "м",
        "t": "т",
        "h": "н",
        "b": "в",
        "A": "А",
        "C": "С",
        "E": "Е",
        "O": "О",
        "P": "Р",
        "X": "Х",
        "Y": "У",
        "K": "К",
        "M": "М",
        "T": "Т",
        "H": "Н",
        "B": "В",
    }
)


def normalize_form(s):
    return s.translate(LAT_TO_CYR)


def count_games_with_forms(cur):
    cur.execute(
        """
        select game, count(*)
        from """
        + TABLE_NAME
        + """
        where text1 is not null
          and text1 ~ '\\[[A-Za-zА-Яа-яЁё]+_[A-Za-zА-Яа-яЁё]+\\]'
        group by game
        order by game
        """
    )
    return cur.fetchall()


def sample_rows(cur, game, limit=5):
    cur.execute(
        """
        select phrase_order, text1
        from """
        + TABLE_NAME
        + """
        where game = %s
          and text1 is not null
          and text1 ~ '\\[[A-Za-zА-Яа-яЁё]+_[A-Za-zА-Яа-яЁё]+\\]'
        order by phrase_order
        limit %s
        """,
        (game, limit),
    )
    return cur.fetchall()


def collect_forms_for_game(cur, game):
    cur.execute(
        """
        select text1
        from """
        + TABLE_NAME
        + """
        where game = %s
          and text1 is not null
          and text1 ~ '\\[[A-Za-zА-Яа-яЁё]+_[A-Za-zА-Яа-яЁё]+\\]'
        """,
        (game,),
    )
    rows = cur.fetchall()
    c = Counter()
    for (t,) in rows:
        for m in BRACKET_FORM_RE.finditer(t):
            c[(normalize_form(m.group(1)), normalize_form(m.group(2)))] += 1
    return c


def truncate(s, max_len=240):
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def ask_choice_for_game(game, samples, counter):
    print("\nИгра:", game)
    print("Примеры строк с паттерном [левая_правая] (до 5):\n")
    for i, (order, text) in enumerate(samples, 1):
        print("---", i, "---", "phrase_order", order)
        print(truncate(text, 320))
        print()

    print("Частые варианты (левая | правая | вхождений):")
    for (l, r), cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:15]:
        print(" ", l, "|", r, "|", cnt)

    print(
        "\nВыбор для всей игры:\n"
        " [l] вставлять ЛЕВУЮ часть\n"
        " [r] вставлять ПРАВУЮ часть\n"
        " [s] пропустить игру\n"
        " [q] выход\n"
        "Примечание: если выбранная часть == 'нет', будет подставлена пустая строка.\n"
    )
    while True:
        choice = input().strip().lower()
        if choice in ("l", "left", "1", "л"):
            return "L"
        if choice in ("r", "right", "2", "п", "р"):
            return "R"
        if choice in ("s", "skip", ""):
            return None
        if choice in ("q", "quit", "exit"):
            return "QUIT"
        print("Не понял ввод, повторите.")


def choose_part(left, right, side):
    if side == "L":
        chosen = left
    else:
        chosen = right
    chosen = normalize_form(chosen)
    if chosen.lower() == "нет":
        return ""
    return chosen


def transform_text(text, side):
    if text is None:
        return None

    def repl(m):
        return choose_part(m.group(1), m.group(2), side)

    return BRACKET_FORM_RE.sub(repl, text)


def apply_updates_for_game(cur, conn, game, side):
    cur.execute(
        """
        select file, line_number, text1
        from """
        + TABLE_NAME
        + """
        where game = %s
          and text1 is not null
          and text1 ~ '\\[[A-Za-zА-Яа-яЁё]+_[A-Za-zА-Яа-яЁё]+\\]'
        """,
        (game,),
    )
    rows = cur.fetchall()
    batch = []
    changed = 0

    def flush(lst):
        if not lst:
            return
        cur.executemany(
            """
            update """
            + TABLE_NAME
            + """
               set text1 = %s
             where game = %s and file = %s and line_number = %s
            """,
            lst,
        )
        lst.clear()

    for file_, line_no, text in rows:
        new_t = transform_text(text, side)
        if new_t == text:
            continue
        changed += 1
        batch.append((new_t, game, file_, line_no))
        if len(batch) >= 500:
            flush(batch)

    flush(batch)
    conn.commit()
    print("Изменено строк в игре:", changed, "|", game)


def main():
    conn_str = os.environ.get("pghost")
    if not conn_str:
        print("Ошибка: переменная окружения pghost не задана")
        sys.exit(1)

    conn = psycopg.connect(conn_str)
    cur = conn.cursor()

    games = count_games_with_forms(cur)
    if not games:
        print("Не найдено игр со скобочными формами вида [левая_правая] в text1.")
        conn.close()
        return

    print("Игр с паттерном [левая_правая]:", len(games))

    for game, cnt in games:
        samples = sample_rows(cur, game)
        forms = collect_forms_for_game(cur, game)
        choice = ask_choice_for_game(game, samples, forms)
        if choice == "QUIT":
            print("Выход.")
            break
        if choice is None:
            print("Игра пропущена:", game)
            continue
        apply_updates_for_game(cur, conn, game, choice)

    conn.close()


if __name__ == "__main__":
    main()

