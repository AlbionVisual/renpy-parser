from __future__ import annotations

import json
import math
import numbers
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, TextIO


def apply_run_dir_defaults(args: Namespace) -> None:
    rd = getattr(args, "run_dir", None)
    if rd is None:
        return
    args.run_dir = Path(rd).expanduser().resolve()
    if getattr(args, "per_column_metrics_csv", None) is None:
        args.per_column_metrics_csv = args.run_dir / "per_column_metrics.csv"


class _Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self.streams:
            n = st.write(s)
            st.flush()
        return n

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def writable(self) -> bool:
        return True


def _jsonable_value(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_jsonable_value(x) for x in v]
    if isinstance(v, Namespace):
        return {k: _jsonable_value(x) for k, x in vars(v).items()}
    return str(v)


def write_run_config_json(run_dir: Path, args: Namespace, extra: dict[str, Any] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    d: dict[str, Any] = {k: _jsonable_value(v) for k, v in vars(args).items()}
    if extra:
        d["extra"] = {k: _jsonable_value(v) for k, v in extra.items()}
    (run_dir / "config.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_json_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_floats(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return obj


def write_metrics_json(run_dir: Path, payload: dict[str, Any]) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        clean = _sanitize_json_floats(payload)
        (run_dir / "metrics.json").write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(
            "write_metrics_json failed",
            run_dir,
            type(exc).__name__,
            str(exc),
        )
        traceback.print_exc()


class RunArtifacts:
    def __init__(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir.expanduser().resolve() if run_dir is not None else None
        self._log_f: TextIO | None = None
        self._old_out: TextIO | None = None
        self._old_err: TextIO | None = None

    def __enter__(self) -> RunArtifacts:
        if self.run_dir is None:
            return self
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_f = open(self.run_dir / "stdout.log", "a", encoding="utf-8", buffering=1)
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        sys.stdout = _Tee(self._old_out, self._log_f)
        sys.stderr = _Tee(self._old_err, self._log_f)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.run_dir is None or self._log_f is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self._old_out
        sys.stderr = self._old_err
        self._log_f.close()
        self._log_f = None
