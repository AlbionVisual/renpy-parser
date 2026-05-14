import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import psycopg

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from text_scripts.ml_scripts.rubert_embeddings import validate_pg_identifier
from text_scripts.ml_scripts.train_rubert_to_music_linear import (
    default_dsn,
    fetch_candidates,
    load_frames,
    music_mode_note,
)


def gameSlug(game):
    h = hashlib.sha1(game.encode("utf-8")).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in game)[:40]
    return safe + "_" + h


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dsn", default=None)
    p.add_argument("--rubert-table", default="text_musics_rubert")
    p.add_argument(
        "--embedding-column",
        default="text_emb_concat4_stride4_max512",
    )
    p.add_argument(
        "--merge-identical-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--music-table", default="music_data")
    p.add_argument("--y-group", choices=("librosa", "jamendo", "all"), default="all")
    p.add_argument("--music-mode", default="min_loss")
    return p.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    validate_pg_identifier(args.rubert_table)
    validate_pg_identifier(args.embedding_column)
    validate_pg_identifier(args.music_table)
    music_mode_note(args.music_mode)
    dsn = args.dsn or default_dsn()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cand_root = out / "candidates"
    cand_root.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn) as conn:
        loaded = load_frames(conn, args)
        if loaded is None:
            sys.exit(2)
        n = len(loaded.rows)
        x_dim = int(loaded.x.shape[1])
        y_dim = int(loaded.y.shape[1])

        row_ids = np.arange(n, dtype=np.int64)
        meta_path = out / "meta.csv"
        with meta_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "game", "phrase_order", "music_id", "path"])
            for i, r in enumerate(loaded.rows):
                w.writerow([i, r.game, r.phrase_order, r.music_id, r.path or ""])

        np.save(out / "x.npy", loaded.x.astype(np.float32, copy=False))
        np.save(out / "y.npy", loaded.y.astype(np.float32, copy=False))
        (out / "y_columns.json").write_text(
            json.dumps(loaded.y_columns, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        games = sorted({r.game for r in loaded.rows})
        cand_index = {}
        for g in games:
            cands = fetch_candidates(conn, args.music_table, loaded.y_columns, [g])
            lst = cands.get(g, [])
            if not lst:
                continue
            ids = np.asarray([t[0] for t in lst], dtype=np.int32)
            mat = np.stack([t[1] for t in lst], axis=0).astype(np.float32, copy=False)
            fn = gameSlug(g) + ".npz"
            np.savez_compressed(cand_root / fn, music_id=ids, y=mat)
            cand_index[g] = fn

        (cand_root / "index.json").write_text(
            json.dumps(cand_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = {
            "format": "dataset_v1",
            "n_rows": n,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "rubert_table": args.rubert_table,
            "embedding_column": args.embedding_column,
            "music_table": args.music_table,
            "y_group": args.y_group,
            "merge_identical_embeddings": bool(args.merge_identical_embeddings),
            "meta_file": "meta.csv",
            "x_file": "x.npy",
            "y_file": "y.npy",
            "y_columns_file": "y_columns.json",
            "candidates_index": "candidates/index.json",
        }
        (out / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        prev = None
        for i, r in enumerate(loaded.rows):
            key = (r.game, r.phrase_order, r.music_id)
            if prev is not None and key < prev:
                print("fatal: rows not sorted by (game, phrase_order, music_id)")
                sys.exit(2)
            prev = key

        print("wrote", str(out))
        print("n_rows", n, "x_dim", x_dim, "y_dim", y_dim, "games", len(games))
        print("candidates_games", len(cand_index))
    sys.exit(0)
