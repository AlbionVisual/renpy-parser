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

from consts import AUDIOS_DIR, TABLE_NAME

load_dotenv()

_AUDIO_EXT = frozenset({".ogg", ".opus", ".mp3", ".wav"})

_MANUAL_PATH_OVERRIDES: dict[tuple[str, str], str] = {
    (
        "Entrance-2.0-pc",
        "audio/mysterious_whisper.mp3",
    ): "audio/desperate_whisper.mp3",
}


def _alnum_only(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def _prefix_unique_audio_match(rels: list[str], token: str) -> str | None:
    ta = _alnum_only(token)
    if len(ta) < 4:
        return None
    hits: list[str] = []
    for r in rels:
        stem_a = _alnum_only(Path(r).stem)
        if stem_a.startswith(ta):
            hits.append(r)
    if not hits:
        return None
    exact = [r for r in hits if _alnum_only(Path(r).stem) == ta]
    if len(exact) == 1:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    return None


def _list_audio_rels(game: str) -> list[str]:
    root = AUDIOS_DIR / game
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXT:
            out.append(p.relative_to(root).as_posix())
    return out


def _build_indexes(rels: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_lower: dict[str, str] = {}
    by_stem: dict[str, list[str]] = {}
    for r in rels:
        low = r.lower()
        if low not in by_lower or len(r) < len(by_lower[low]):
            by_lower[low] = r
        stem = Path(r).stem.lower()
        by_stem.setdefault(stem, []).append(r)
    return by_lower, by_stem


def _resolve_path(
    by_lower: dict[str, str],
    by_stem: dict[str, list[str]],
    music_path: str,
    rels: list[str],
    game: str,
) -> str | None:
    s0 = (music_path or "").strip().replace("\\", "/").lstrip("/")
    if not s0 or s0 == "NO_MUSIC":
        return None
    ovr = _MANUAL_PATH_OVERRIDES.get((game, s0))
    if ovr is not None and ovr.lower() in by_lower:
        return by_lower[ovr.lower()]
    s = s0
    if s.upper().startswith("UNKNOWN:"):
        s = s.split(":", 1)[1].strip().lstrip("/")
    if not s:
        return None
    if s.lower() in by_lower:
        return by_lower[s.lower()]
    cand = "audio/" + s if not s.lower().startswith("audio/") else s
    if cand.lower() in by_lower:
        return by_lower[cand.lower()]
    base = Path(s).name.lower()
    name_matches = [r for r in by_lower.values() if Path(r).name.lower() == base]
    if len(name_matches) == 1:
        return name_matches[0]
    stem = Path(s).stem.lower()
    stem_hits = by_stem.get(stem)
    if stem_hits and len(stem_hits) == 1:
        return stem_hits[0]
    if stem_hits:
        s_low = s.lower()
        best = None
        best_len = 10**9
        for h in stem_hits:
            hl = h.lower()
            if s_low in hl or hl.endswith(s_low) or s_low.endswith(hl):
                if len(h) < best_len:
                    best = h
                    best_len = len(h)
        if best is not None:
            return best
    guess = _prefix_unique_audio_match(rels, s)
    if guess is not None:
        return guess
    return None


def _distinct_bad_pairs(cur, table: str) -> list[tuple[str, str]]:
    cur.execute(
        f"""
        select distinct game, music_path
        from {table}
        where music is null
          and coalesce(trim(music_path), '') <> 'NO_MUSIC'
        """,
    )
    return [(str(a), str(b)) for a, b in cur.fetchall()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Сопоставляет music_path в БД с реальными файлами под audios/<game>/ "
        "и обновляет music_path на канонический относительный путь. "
        "Затем добавляет недостающие строки в music_data и вызывает sync_columns(). "
        "По умолчанию только предпросмотр; для записи укажите --apply.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="выполнить UPDATE в БД, insert в music_data и sync_columns()",
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

    by_game: dict[str, tuple[dict[str, str], dict[str, list[str]]]] = {}

    def indexes_for(game: str):
        if game not in by_game:
            rels = _list_audio_rels(game)
            by_game[game] = _build_indexes(rels)
        return by_game[game]

    changes: list[tuple[str, str, str]] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            pairs = _distinct_bad_pairs(cur, TABLE_NAME)
        for game, old_path in pairs:
            by_lower, by_stem = indexes_for(game)
            if not by_lower:
                print("skip no audios dir:", game, AUDIOS_DIR / game)
                continue
            rels = _list_audio_rels(game)
            new_path = _resolve_path(by_lower, by_stem, old_path, rels, game)
            if new_path is None:
                print("no match:", game, repr(old_path))
                continue
            old_n = old_path.strip().replace("\\", "/").lstrip("/").lower()
            new_n = new_path.lower()
            if new_n == old_n:
                continue
            changes.append((game, old_path, new_path))

    print("planned updates:", len(changes))
    for game, old_path, new_path in changes[:50]:
        print("UPDATE", game, repr(old_path), "->", repr(new_path))
    if len(changes) > 50:
        print("... and", len(changes) - 50, "more")

    if not args.apply:
        print("preview only. Pass --apply to write DB and run sync_columns().")
        return 0

    import importlib.util

    _pfd_path = _REPO_ROOT / "music_scripts" / "parse_folder_to_db.py"
    _spec = importlib.util.spec_from_file_location("parse_folder_to_db", _pfd_path)
    _pfd = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_pfd)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for game, old_path, new_path in changes:
                cur.execute(
                    f"""
                    update {TABLE_NAME}
                    set music_path = %s
                    where game = %s and music_path = %s and music is null
                    """,
                    (new_path, game, old_path),
                )
                if cur.rowcount:
                    print("updated rows:", cur.rowcount, game, old_path, "->", new_path)
            cur.execute(
                """
                insert into music_data (game, path)
                select distinct t.game, trim(t.music_path)
                from {tbl} t
                where t.music_path is not null
                  and t.music_path <> 'NO_MUSIC'
                  and trim(t.music_path) not like 'UNKNOWN:%%'
                  and not exists (
                    select 1 from music_data m
                    where m.game = t.game and m.path = trim(t.music_path)
                  )
                """.format(
                    tbl=TABLE_NAME,
                ),
            )
        conn.commit()

    _pfd.sync_columns()
    print("sync_columns() done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
