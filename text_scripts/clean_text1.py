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


PLACEHOLDER_BODY = re.compile(
    r"\[\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\]",
    re.UNICODE,
)

BRACKET_FORM_RE = re.compile(r"\[([A-Za-zА-Яа-яЁё]+)_([A-Za-zА-Яа-яЁё]+)\]")

WHOLE_BRACKET_LINE_RE = re.compile(r"^\s*\[(.*)\]\s*$", re.DOTALL)
CYRILLIC_BRACKET_TEXT_RE = re.compile(r"\[([^\[\]]*[А-Яа-яЁё][^\[\]]*)\]")

ALLOWED_TEXT_RE = re.compile(
    r"[^A-Za-zА-Яа-яЁё0-9\s\.\,\!\?\;\:\-\—–…«»\"„\(\)/%№'_]",
    re.UNICODE,
)

LAT_CYR_DIG_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


BAD_ROWS_SQL = """(
      text1 ~ '\\.{4,}'
   or text1 ~ '\\s{2,}'
   or text1 ~ '\\[\\s*[A-Za-z_][A-Za-z0-9_]*\\s*\\]'
   or text1 ~ '\\{[^}]*\\}'
   or text1 ~ '<[^<>]+>'
   or text1 ~ '\\\\'
   or text1 !~ '^[А-Яа-яЁё0-9[:space:]''"_\\.\\,\\!\\?\\;\\:\\-\\—–…«»""„()/%%№]+$'
   )"""


def count_bad_rows_by_game(cur):
    cur.execute(
        """
        select game, count(*)
        from """
        + TABLE_NAME
        + """
        where text1 is not null
          and """
        + BAD_ROWS_SQL
        + """
        group by game
        order by game
        """
    )
    return dict(cur.fetchall())


def truncate(s, max_len=240):
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


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


def sample_form_rows(cur, game, limit=5):
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
    chosen = left if side == "L" else right
    chosen = normalize_form(chosen)
    if chosen.lower() == "нет":
        return ""
    return chosen


def apply_bracket_forms(text, side):
    if text is None or side is None:
        return text

    def repl(m):
        return choose_part(m.group(1), m.group(2), side)

    return BRACKET_FORM_RE.sub(repl, text)


def collect_placeholder_names_and_examples(rows, max_examples_per_name=5):
    c = Counter()
    examples = {}
    for text, in rows:
        if not text:
            continue
        for m in PLACEHOLDER_BODY.finditer(text):
            name = m.group(1)
            c[name] += 1
            ex = examples.get(name)
            if ex is None:
                examples[name] = [text]
            elif len(ex) < max_examples_per_name:
                ex.append(text)
    return c, examples


def prompt_placeholder_replacements(counter_by_name, examples_by_name, max_examples=5):
    if not counter_by_name:
        print("Плейсхолдеров вида [latin_name] не найдено.")
        return {}
    print("\nНайдены плейсхолдеры (внутреннее имя без скобок). Для каждого укажи замену.")
    print(
        "Пустая строка — заменить на пустоту. [s] — не трогать. [q] — выйти без применения.\n"
        "Если ввести 'auto', будет применено правило (для некоторых плейсхолдеров).\n"
    )
    mapping = {}
    items = sorted(counter_by_name.items(),
                   key=lambda x: (-x[1], x[0].lower()))
    for name, cnt in items:
        print("---")
        print("Имя:", name, "| вхождений:", cnt)
        ex = examples_by_name.get(name) or []
        if ex:
            print("Примеры (до " + str(max_examples) + "):")
            for i, t in enumerate(ex[:max_examples], 1):
                print(" ", i, truncate(t, 320))
        else:
            print("Примеры: (нет)")
        print("Замена (или s=skip, q=quit):")
        line = input()
        if line.strip().lower() == "q":
            return None
        if line.strip().lower() == "s":
            mapping[name] = None
            continue
        if line.strip().lower() == "auto":
            mapping[name] = "__AUTO__"
        else:
            mapping[name] = line
    return mapping


def apply_placeholder_replacements(text, mapping):
    if text is None:
        return None
    t = text

    def apply_mestgl3_auto(s):
        pat = re.compile(r"\[\s*mestGL3\s*\]", re.IGNORECASE)

        def repl(m):
            i = m.start()
            prev = s[i - 1] if i > 0 else ""
            return "а" if prev == "л" else "ла"

        return pat.sub(repl, s)

    for name, repl in mapping.items():
        if repl is None:
            continue
        if repl == "__AUTO__" and name.lower() == "mestgl3":
            t = apply_mestgl3_auto(t)
            continue
        if repl == "__AUTO__":
            continue
        pat = r"\[\s*" + re.escape(name) + r"\s*\]"
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t


def clean_pipeline(text, placeholder_mapping, bracket_side=None):
    if text is None:
        return None
    t = text
    t = apply_bracket_forms(t, bracket_side)
    t = re.sub(r"\\[nrt]", " ", t)
    t = t.replace(r"\"", '"')
    t = t.replace(r"\\", "")
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\.{4,}", "…", t)
    t = apply_placeholder_replacements(t, placeholder_mapping)
    t = re.sub(r"\{[^}]*\}", "", t)

    m = WHOLE_BRACKET_LINE_RE.match(t)
    if m:
        inner = m.group(1)
        has_latin = re.search(r"[A-Za-z]", inner) is not None
        has_space = re.search(r"\s", inner) is not None
        if (not has_latin) or has_space:
            t = inner

    def unbracket_cyr(m):
        inner = m.group(1)
        if "_" in inner:
            return m.group(0)
        if "." in inner:
            return m.group(0)
        return inner
    t = CYRILLIC_BRACKET_TEXT_RE.sub(unbracket_cyr, t)

    letters = sum(1 for ch in t if ch.isalpha())
    if letters > 0:
        lat_cyr_digits = len(LAT_CYR_DIG_RE.findall(t))
        if lat_cyr_digits / letters < 0.5:
            return ""

    t = ALLOWED_TEXT_RE.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t


def fetch_all_text1(cur):
    cur.execute(
        """
        select game, file, line_number, text1
        from """
        + TABLE_NAME
        + """
        where text1 is not null
        """
    )
    return cur.fetchall()


def print_stats(title, counts_by_game):
    print("\n=== ", title, " ===")
    if not counts_by_game:
        print("(нет строк, попадающих под фильтр)")
        return
    total = sum(counts_by_game.values())
    print("Всего плохих строк:", total)
    for game, n in sorted(counts_by_game.items()):
        print(" ", game, "|", n)


def main():
    conn_str = os.environ.get("pghost")
    if not conn_str:
        print("Ошибка: переменная окружения pghost не задана")
        sys.exit(1)

    conn = psycopg.connect(conn_str)
    cur = conn.cursor()

    before = count_bad_rows_by_game(cur)
    print_stats('ДО очистки (фильтр "плохих" строк как в запросе)', before)

    bracket_choice_by_game = {}
    games_with_forms = count_games_with_forms(cur)
    if games_with_forms:
        print("\nНайдено игр с формами [левая_правая]:", len(games_with_forms))
        for game, _cnt in games_with_forms:
            samples = sample_form_rows(cur, game, limit=5)
            forms = collect_forms_for_game(cur, game)
            choice = ask_choice_for_game(game, samples, forms)
            if choice == "QUIT":
                print("Выход без изменений.")
                conn.close()
                return
            if choice is None:
                continue
            bracket_choice_by_game[game] = choice

    cur.execute(
        "select text1 from " + TABLE_NAME + " where text1 is not null"
    )
    rows = cur.fetchall()
    ctr, examples = collect_placeholder_names_and_examples(rows, max_examples_per_name=5)

    placeholder_mapping = prompt_placeholder_replacements(ctr, examples, max_examples=5)
    if placeholder_mapping is None:
        print("Выход без изменений.")
        conn.close()
        return

    all_rows = fetch_all_text1(cur)
    updated = 0
    unchanged = 0
    batch = []

    def flush_batch(cur, lst):
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

    try:
        for game, file_, line_no, text in all_rows:
            side = bracket_choice_by_game.get(game)
            new_t = clean_pipeline(text, placeholder_mapping, bracket_side=side)
            if new_t == text:
                unchanged += 1
                continue
            batch.append((new_t, game, file_, line_no))
            updated += 1
            if len(batch) >= 500:
                flush_batch(cur, batch)
        flush_batch(cur, batch)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Ошибка, откат транзакции:", e)
        raise

    print("\nСтрок всего:", len(all_rows))
    print("Изменено:", updated)
    print("Без изменений поля text1:", unchanged)

    after = count_bad_rows_by_game(cur)
    print_stats('ПОСЛЕ очистки', after)

    print("\n--- Сводка по играм (было -> стало) ---")
    all_games = sorted(set(before) | set(after))
    s1 = 0
    s2 = 0
    for game in all_games:
        b = before.get(game, 0)
        a = after.get(game, 0)
        s1 += b - a
        s2 += a
        print(game, "|", b, "->", a)
    print("\nСуммарное количество удаленных строк:", s1)
    print("Суммарное количество оставшихся строк:", s2)

    conn.close()


if __name__ == "__main__":
    main()
