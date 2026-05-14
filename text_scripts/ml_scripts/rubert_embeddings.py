from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal, NamedTuple, Sequence, cast

import numpy as np
import psycopg
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

Pooling = Literal["mean", "cls"]
MergeOverlap = Literal["mean", "last", "first"]


class TextMusicsRubertRow(NamedTuple):
    game: str
    phrase_order: int
    text1: str | None
    music: int | None


def validate_pg_identifier(name: str) -> str:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
        raise ValueError("invalid SQL identifier: " + repr(name))
    return name


def default_dsn() -> str:
    load_dotenv()
    dsn = os.environ.get("pghost")
    if not dsn:
        raise ValueError("environment variable pghost is not set")
    return dsn


def ensure_embedding_column(
    dsn: str,
    *,
    table: str = "text_musics_rubert",
    column: str = "text_analized",
) -> None:
    t = validate_pg_identifier(table)
    c = validate_pg_identifier(column)
    sql = "alter table " + t + " add column if not exists " + c + " real[]"
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def fetch_text_musics_rubert_ordered(
    dsn: str,
    *,
    table: str = "text_musics_rubert",
    only_missing_embeddings: bool = True,
    embedding_column: str = "text_analized",
) -> list[TextMusicsRubertRow]:
    t = validate_pg_identifier(table)
    c = validate_pg_identifier(embedding_column)
    missing_clause = " and " + c + " is null" if only_missing_embeddings else ""
    q = (
        "select game, phrase_order, text1, music from "
        + t
        + " where text1 is not null and btrim(text1) <> '' "
        + missing_clause
        + " order by game, phrase_order"
    )
    out: list[TextMusicsRubertRow] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            for row in cur.fetchall():
                g, po, t1, mu = row
                out.append(
                    TextMusicsRubertRow(
                        str(g),
                        int(po),
                        None if t1 is None else str(t1),
                        None if mu is None else int(mu),
                    ),
                )
    return out


def update_text_analized(
    dsn: str,
    vectors_by_key: dict[tuple[str, int], np.ndarray],
    *,
    table: str = "text_musics_rubert",
    column: str = "text_analized",
    show_progress: bool = False,
    progress_every: int = 500,
) -> int:
    t = validate_pg_identifier(table)
    c = validate_pg_identifier(column)
    ensure_sql = "alter table " + t + " add column if not exists " + c + " real[]"
    sql = (
        "update "
        + t
        + " set "
        + c
        + " = %s::real[] "
        "where game = %s and phrase_order = %s"
    )
    total = len(vectors_by_key)
    done = 0
    pe = max(1, int(progress_every))
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(ensure_sql)
            for (game, po), vec in vectors_by_key.items():
                flat = np.asarray(vec, dtype=np.float64).tolist()
                cur.execute(sql, (flat, game, po))
                done += 1
                if show_progress and (done == 1 or done % pe == 0 or done == total):
                    print("db update", done, "/", total, flush=True)
        conn.commit()
    return total


def pick_device(pref: str | None) -> torch.device:
    if pref in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden.shape).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def load_rubert(
    model_name_or_path: str,
    device: torch.device,
    local_files_only: bool = False,
) -> tuple[torch.nn.Module, object]:
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model, tok


def load_rubert_tokenizer_only(
    model_name_or_path: str,
    *,
    local_files_only: bool = False,
):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )


def encode_batches(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: Pooling,
    use_amp_bf16: bool,
    use_amp_fp16: bool,
    show_progress: bool = False,
) -> Iterator[np.ndarray]:
    n = len(texts)
    total_batches = (n + batch_size - 1) // batch_size if batch_size > 0 and n else 0
    batch_idx = 0
    for start in range(0, n, batch_size):
        chunk = list(texts[start : start + batch_size])
        if not chunk:
            continue
        batch_idx += 1
        if show_progress:
            end_excl = min(start + batch_size, n)
            print(
                "rubert batch",
                batch_idx,
                "/",
                total_batches,
                "examples",
                start,
                "..",
                end_excl,
                flush=True,
            )
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            if device.type == "cuda" and use_amp_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**enc)
            elif device.type == "cuda" and use_amp_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**enc)
            else:
                out = model(**enc)
        hs = out.last_hidden_state.float()
        if pooling == "mean":
            vec = mean_pool(hs, enc["attention_mask"])
        else:
            vec = hs[:, 0]
        yield vec.cpu().numpy().astype(np.float32, copy=False)


def encode_all(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: Pooling,
    use_amp_bf16: bool,
    use_amp_fp16: bool,
    show_progress: bool = False,
) -> np.ndarray:
    if len(texts) == 0:
        hidden = int(getattr(model.config, "hidden_size", 768))
        return np.zeros((0, hidden), dtype=np.float32)
    parts = list(
        encode_batches(
            model,
            tokenizer,
            texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
            use_amp_bf16=use_amp_bf16,
            use_amp_fp16=use_amp_fp16,
            show_progress=show_progress,
        ),
    )
    return cast(np.ndarray, np.concatenate(parts, axis=0))


def encode_all_overlapping(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    keys: Sequence[tuple[str, int]],
    *,
    device: torch.device,
    batch_size: int,
    stride: int,
    max_length: int,
    pooling: Pooling,
    use_amp_bf16: bool,
    use_amp_fp16: bool,
    merge: MergeOverlap,
    show_progress: bool = False,
) -> dict[tuple[str, int], np.ndarray]:
    n = len(texts)
    if n != len(keys):
        raise ValueError("texts and keys length mismatch")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if n == 0:
        return {}
    from collections import defaultdict

    total_windows = (n + stride - 1) // stride
    window_idx = 0
    acc: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for start in range(0, n, stride):
        end = min(start + batch_size, n)
        chunk_t = texts[start:end]
        chunk_k = keys[start:end]
        if not chunk_t:
            continue
        window_idx += 1
        if show_progress:
            print(
                "rubert overlap window",
                window_idx,
                "/",
                total_windows,
                "examples",
                start,
                "..",
                end,
                flush=True,
            )
        vecs = encode_all(
            model,
            tokenizer,
            chunk_t,
            device=device,
            batch_size=len(chunk_t),
            max_length=max_length,
            pooling=pooling,
            use_amp_bf16=use_amp_bf16,
            use_amp_fp16=use_amp_fp16,
            show_progress=False,
        )
        for i, k in enumerate(chunk_k):
            acc[k].append(vecs[i])
    out: dict[tuple[str, int], np.ndarray] = {}
    for k, vs in acc.items():
        stack = np.stack(vs, axis=0)
        if merge == "mean":
            out[k] = np.mean(stack, axis=0).astype(np.float32, copy=False)
        elif merge == "last":
            out[k] = vs[-1].astype(np.float32, copy=False)
        else:
            out[k] = vs[0].astype(np.float32, copy=False)
    return out


def iter_jsonl_texts(path: Path, field: str) -> Iterable[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield str(row[field])


def read_lines_txt(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").split("\n")
    return [ln.rstrip("\r") for ln in raw]

