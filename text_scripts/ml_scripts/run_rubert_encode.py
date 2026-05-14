from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

_ml = Path(__file__).resolve().parent
_text_scripts = _ml.parent
_repo = _text_scripts.parent
for _p in (_repo, _text_scripts):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch

from text_scripts.ml_scripts.db_consts import TEXT_MUSICS_RUBERT_TABLE
from ml_scripts.rubert_embeddings import (
    default_dsn,
    encode_all,
    ensure_embedding_column,
    fetch_text_musics_rubert_ordered,
    iter_jsonl_texts,
    load_rubert,
    load_rubert_tokenizer_only,
    pick_device,
    read_lines_txt,
    update_text_analized,
)


def _token_count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=True))


def _truncate_to_max_tokens(tok, text: str, max_tokens: int) -> str:
    if not text:
        return text
    ids = tok.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
    )
    return tok.decode(ids, skip_special_tokens=True).strip()


def build_windows_by_token_budget(
    tok,
    texts: list[str],
    keys: list[tuple[str, int]],
    *,
    max_tokens: int,
    group_by_game: bool,
    show_progress: bool,
) -> tuple[list[str], list[list[tuple[str, int]]]]:
    window_texts: list[str] = []
    window_keys: list[list[tuple[str, int]]] = []

    def add_window(klist: list[tuple[str, int]], tlist: list[str]) -> None:
        if not klist:
            return
        window_keys.append(list(klist))
        window_texts.append("\n".join(tlist))

    def flush_segment(
        buf_k: list[tuple[str, int]],
        buf_t: list[str],
    ) -> None:
        if not buf_k:
            return
        add_window(buf_k, buf_t)

    def iter_segments():
        n = len(texts)
        if not n:
            return
        if not group_by_game:
            yield 0, n
            return
        start = 0
        while start < n:
            g = keys[start][0]
            end = start + 1
            while end < n and keys[end][0] == g:
                end += 1
            yield start, end
            start = end

    for seg_start, seg_end in iter_segments():
        buf_k: list[tuple[str, int]] = []
        buf_t: list[str] = []
        for idx in range(seg_start, seg_end):
            line = texts[idx]
            key = keys[idx]
            trial = "\n".join(buf_t + [line]) if buf_t else line
            if _token_count(tok, trial) <= max_tokens:
                buf_k.append(key)
                buf_t.append(line)
                continue
            if not buf_k:
                clipped = _truncate_to_max_tokens(tok, line, max_tokens)
                add_window([key], [clipped])
                continue
            flush_segment(buf_k, buf_t)
            buf_k = []
            buf_t = []
            one = line
            if _token_count(tok, one) <= max_tokens:
                buf_k = [key]
                buf_t = [line]
            else:
                clipped = _truncate_to_max_tokens(tok, line, max_tokens)
                add_window([key], [clipped])
        flush_segment(buf_k, buf_t)

    if show_progress:
        print(
            "concat-by-tokens budget",
            max_tokens,
            "windows",
            len(window_texts),
            "from",
            len(keys),
            "rows",
            flush=True,
        )
    return window_texts, window_keys


def main() -> int:
    p = argparse.ArgumentParser(
        description="RuBERT pooled embeddings: файлы, jsonl или БД; опционально запись text_analized.",
    )
    p.add_argument(
        "--model",
        default="DeepPavlov/rubert-base-cased",
        help="HF id или локальная папка",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps | cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--concat-lines",
        type=int,
        default=1,
        help="окно склейки: сколько строк объединить в один текст; 1 = без склейки",
    )
    p.add_argument(
        "--concat-by-tokens",
        action="store_true",
        help="склеивать подряд идущие строки в одно окно по числу токенов (тот же счёт, "
        "что и tokenizer.encode(..., add_special_tokens=True)), пока длина <= --max-length; "
        "требует --concat-lines 1; с --from-db границы только внутри одной game",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=0,
        help="шаг окна: если concat-lines>1, то сдвиг окна склейки в строках (0 = 1); иначе шаг окон для overlap режима (0 = batch-size)",
    )
    p.add_argument(
        "--overlap-merge",
        choices=("mean", "last", "first"),
        default="first",
        help="как сливать векторы для одной строки из пересекающихся окон",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    p.add_argument("--amp-fp16", action="store_true")
    p.add_argument("--amp-bf16", action="store_true")
    p.add_argument("--torch-compile", action="store_true")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-txt", type=Path)
    src.add_argument("--input-jsonl", type=Path)
    src.add_argument(
        "--from-db",
        action="store_true",
        help="читать строки из text_musics_rubert (text1 не пустой), порядок game, phrase_order",
    )
    p.add_argument(
        "--reencode-all",
        action="store_true",
        help="со --from-db: все строки с text1, даже с уже заполненным text_analized (по умолчанию только text_analized is null)",
    )

    p.add_argument("--text-field", default="text1")
    p.add_argument("--output-npy", type=Path, default=None)
    p.add_argument("--meta-json", type=Path, default=None)
    p.add_argument(
        "--write-db",
        action="store_true",
        help="записать эмбеддинги в колонку БД (только с --from-db)",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help="строка подключения; иначе env pghost",
    )
    p.add_argument(
        "--db-table",
        default=TEXT_MUSICS_RUBERT_TABLE,
        help="имя таблицы с text1 и text_analized",
    )
    p.add_argument(
        "--db-column",
        default="text_analized",
        help="колонка real[] для записи эмбеддингов (создастся автоматически при --write-db)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="не печатать прогресс батчей/окон и апдейтов БД",
    )
    p.add_argument(
        "--db-progress-every",
        type=int,
        default=500,
        help="логировать каждые N строк при записи в БД (если прогресс включён)",
    )

    args = p.parse_args()

    if args.batch_size < 1:
        print("batch-size must be >= 1")
        return 2

    if args.concat_lines < 1:
        print("concat-lines must be >= 1")
        return 2

    if args.concat_by_tokens:
        if args.concat_lines != 1:
            print("--concat-by-tokens: укажите --concat-lines 1")
            return 2
        if args.stride != 0:
            print("--concat-by-tokens: не используйте --stride (жадная упаковка по токенам)")
            return 2
        stride = 0
    elif args.concat_lines > 1:
        stride = args.stride if args.stride > 0 else 1
    else:
        stride = args.stride if args.stride > 0 else args.batch_size
        if stride > args.batch_size:
            print("stride must be <= batch-size (иначе часть строк ни разу не попадёт в окно)")
            return 2

    if args.write_db and not args.from_db:
        print("--write-db только вместе с --from-db")
        return 2

    if args.reencode_all and not args.from_db:
        print("--reencode-all имеет смысл только с --from-db")
        return 2

    if args.output_npy is None and not args.write_db:
        print("нужен --output-npy и/или --write-db")
        return 2

    dsn = None
    if args.from_db:
        dsn = args.dsn or default_dsn()

    texts: list[str] = []
    keys: list[tuple[str, int]] = []

    show_progress = not args.no_progress

    if args.from_db:
        if not args.reencode_all:
            ensure_embedding_column(
                dsn,
                table=args.db_table,
                column=args.db_column,
            )
        rows = fetch_text_musics_rubert_ordered(
            dsn,
            table=args.db_table,
            only_missing_embeddings=not args.reencode_all,
            embedding_column=args.db_column,
        )
        if show_progress:
            mode = "all rows (reencode-all)" if args.reencode_all else ("only " + args.db_column + " is null")
            print(
                "loaded from db",
                len(rows),
                "rows table",
                args.db_table,
                "filter",
                mode,
                flush=True,
            )
        for r in rows:
            t1 = r.text1 or ""
            texts.append(t1)
            keys.append((r.game, r.phrase_order))
    elif args.input_txt is not None:
        texts = read_lines_txt(args.input_txt)
        keys = [("_row", i) for i in range(len(texts))]
        if show_progress:
            print("loaded lines from file", len(texts), flush=True)
    else:
        texts = list(iter_jsonl_texts(args.input_jsonl, args.text_field))
        keys = [("_row", i) for i in range(len(texts))]
        if show_progress:
            print("loaded lines from jsonl", len(texts), flush=True)

    orig_keys = list(keys)
    window_texts: list[str] = []
    window_keys: list[list[tuple[str, int]]] = []

    if args.concat_by_tokens and texts:
        tok_pre = load_rubert_tokenizer_only(
            args.model,
            local_files_only=args.local_files_only,
        )
        window_texts, window_keys = build_windows_by_token_budget(
            tok_pre,
            texts,
            keys,
            max_tokens=args.max_length,
            group_by_game=args.from_db,
            show_progress=show_progress,
        )
    elif args.concat_lines > 1 and texts:
        def add_window(klist: list[tuple[str, int]], tlist: list[str]) -> None:
            if not klist:
                return
            window_keys.append(list(klist))
            window_texts.append("\n".join(tlist))

        if args.from_db:
            start = 0
            n = len(texts)
            while start < n:
                g = keys[start][0]
                end = start
                while end < n and keys[end][0] == g:
                    end += 1
                i = start
                while i < end:
                    j = min(i + args.concat_lines, end)
                    add_window(keys[i:j], texts[i:j])
                    i += stride
                start = end
        else:
            i = 0
            n = len(texts)
            while i < n:
                j = min(i + args.concat_lines, n)
                add_window(keys[i:j], texts[i:j])
                i += stride

        if show_progress:
            print(
                "concat-lines",
                args.concat_lines,
                "stride",
                stride,
                "windows",
                len(window_texts),
                "from",
                len(orig_keys),
                "rows",
                flush=True,
            )
    else:
        window_texts = list(texts)
        window_keys = [[k] for k in keys]

    t_load = perf_counter()
    device = pick_device(None if args.device == "auto" else args.device)
    model, tok = load_rubert(
        args.model,
        device,
        local_files_only=args.local_files_only,
    )
    load_s = perf_counter() - t_load

    if args.torch_compile:
        model = torch.compile(model)

    if show_progress and texts:
        print(
            "encoding",
            len(window_texts),
            "texts",
            "stride",
            stride,
            "concat-by-tokens",
            args.concat_by_tokens,
            "batch-size",
            args.batch_size,
            flush=True,
        )

    t_enc = perf_counter()
    if len(window_texts) == 0:
        merged = {}
    else:
        arr_windows = encode_all(
            model,
            tok,
            window_texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            use_amp_bf16=args.amp_bf16,
            use_amp_fp16=args.amp_fp16,
            show_progress=show_progress,
        )
        from collections import defaultdict

        acc: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for wi, klist in enumerate(window_keys):
            vec = arr_windows[wi]
            for k in klist:
                acc[k].append(vec)

        merged = {}
        for k, vs in acc.items():
            if len(vs) == 1:
                merged[k] = vs[0]
            elif args.overlap_merge == "last":
                merged[k] = vs[-1]
            elif args.overlap_merge == "mean":
                merged[k] = np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32, copy=False)
            else:
                merged[k] = vs[0]
    enc_s = perf_counter() - t_enc

    if args.write_db:
        nup = update_text_analized(
            dsn,
            merged,
            table=args.db_table,
            column=args.db_column,
            show_progress=show_progress,
            progress_every=args.db_progress_every,
        )
        print("updated rows", nup, "in", args.db_table, "column", args.db_column)

    if args.output_npy is not None:
        order = [merged[k] for k in orig_keys]
        if order:
            arr = np.stack(order, axis=0)
        else:
            h = int(getattr(model.config, "hidden_size", 768))
            arr = np.zeros((0, h), dtype=np.float32)
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_npy, arr.astype(np.float32, copy=False))
        print("saved", args.output_npy, "shape", arr.shape)

    print("device", device)
    print("rows", len(orig_keys), "merged_keys", len(merged))
    print("load_model_s", round(load_s, 3), "encode_s", round(enc_s, 3))

    if args.meta_json is not None:
        meta = {
            "model": args.model,
            "local_files_only": args.local_files_only,
            "device": str(device),
            "batch_size": args.batch_size,
            "stride": stride,
            "concat_by_tokens": args.concat_by_tokens,
            "overlap_merge": args.overlap_merge,
            "max_length": args.max_length,
            "pooling": args.pooling,
            "amp_fp16": args.amp_fp16,
            "amp_bf16": args.amp_bf16,
            "torch_compile": args.torch_compile,
            "n_rows": len(texts),
            "from_db": args.from_db,
            "write_db": args.write_db,
            "reencode_all": args.reencode_all,
            "no_progress": args.no_progress,
            "load_model_s": load_s,
            "encode_s": enc_s,
        }
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)
        args.meta_json.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("meta", args.meta_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
