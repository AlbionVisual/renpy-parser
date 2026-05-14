from __future__ import annotations
import argparse
import sys
from pathlib import Path

import psycopg
from psycopg import sql

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from text_scripts.ml_scripts.db_consts import TEXT_MUSICS_RUBERT_TABLE
from text_scripts.ml_scripts.rubert_embeddings import validate_pg_identifier
from text_scripts.ml_scripts.train_rubert_to_music_linear import default_dsn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-game phrase_order stats (joined rubert + music_data).")
    p.add_argument("--dsn", default=None)
    p.add_argument("--rubert-table", default=TEXT_MUSICS_RUBERT_TABLE)
    p.add_argument("--embedding-column", default="text_analized")
    p.add_argument("--music-table", default="music_data")
    p.add_argument("--limit-games", type=int, default=25,
                   help="сколько игр печатать (по убыванию числа строк)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    validate_pg_identifier(args.rubert_table)
    validate_pg_identifier(args.embedding_column)
    validate_pg_identifier(args.music_table)
    emb = args.embedding_column
    dsn = args.dsn or default_dsn()
    q_base = sql.SQL(
        """
        select md.game::text,
               count(*)::bigint as n,
               min(t.phrase_order)::int as lo,
               max(t.phrase_order)::int as hi
        from {rubert} t
        join {music} md on t.music = md.id
        where t.{emb} is not null and t.music is not null
        group by md.game
        order by n desc
        """
    ).format(
        rubert=sql.Identifier(args.rubert_table),
        music=sql.Identifier(args.music_table),
        emb=sql.Identifier(emb),
    )
    q = sql.SQL("{} limit {}").format(
        q_base, sql.Literal(int(args.limit_games)))
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
    print("games_shown", len(rows), "limit", args.limit_games)
    print("game", "n_rows", "phrase_lo", "phrase_hi",
          "span", "span_minus_n", "note")
    gaps = 0
    for game, n, lo, hi in rows:
        span = hi - lo + 1
        diff = int(span - int(n))
        note = ""
        if diff != 0:
            note = "deleted_or_missing_phrase_indices_in_range"
            gaps += 1
        print(game, int(n), lo, hi, span, diff, note)
    print("games_with_span_neq_count", gaps, "of", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
