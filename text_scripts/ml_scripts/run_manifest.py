from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _has_success_artifacts(output_dir: Path) -> bool:
    return (output_dir / "metrics.json").exists() or (output_dir / "per_column_metrics.csv").exists()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run training jobs from manifest.jsonl with resume.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base-run-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    base_run = args.base_run_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    rows = _read_manifest(manifest_path)
    for r in rows:
        if str(r.get("status", "")) == "running":
            od = Path(str(r["output_dir"])).expanduser().resolve()
            if not _has_success_artifacts(od):
                r["status"] = "failed"
    _write_manifest(manifest_path, rows)

    pick_status = ("pending", "failed") if args.resume else ("pending",)
    idx: int | None = None
    for i, r in enumerate(rows):
        if str(r.get("status", "")) in pick_status:
            idx = i
            break
    if idx is None:
        print("manifest: no jobs in status", pick_status, flush=True)
        return 0

    row = rows[idx]
    run_id = str(row["run_id"])
    script_rel = str(row["script"])
    argv_extra = list(row.get("args") or [])
    out_dir = Path(str(row.get("output_dir") or (
        base_run / run_id))).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    row["output_dir"] = str(out_dir)
    row["status"] = "running"
    _write_manifest(manifest_path, rows)

    script_path = _repo_root() / script_rel
    if not script_path.is_file():
        print("fatal: script not found", str(script_path), flush=True)
        row["status"] = "failed"
        _write_manifest(manifest_path, rows)
        return 2

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(script_path),
        *argv_extra,
        "--dataset-dir",
        str(dataset_dir),
        "--run-dir",
        str(out_dir),
    ]
    if "seed" in row and "--seed" not in argv_extra:
        cmd.extend(["--seed", str(int(row["seed"]))])
    if "held_out_games_seed" in row and "--held-out-games-seed" not in argv_extra:
        cmd.extend(["--held-out-games-seed",
                   str(int(row["held_out_games_seed"]))])

    print("manifest run", run_id, flush=True)
    print("cmd", " ".join(cmd), flush=True)
    if args.dry_run:
        row["status"] = "pending"
        _write_manifest(manifest_path, rows)
        return 0

    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    rc = subprocess.run(cmd, cwd=str(_repo_root()), env=child_env).returncode
    ok = rc == 0 and _has_success_artifacts(out_dir)
    row["status"] = "done" if ok else "failed"
    _write_manifest(manifest_path, rows)
    print("finished", run_id, "status",
          row["status"], "returncode", rc, flush=True)
    if rc != 0:
        print(
            "hint: train subprocess exited non-zero; see traceback above and",
            str(out_dir / "stdout.log"),
            flush=True,
        )
    if rc == 0 and not _has_success_artifacts(out_dir):
        print(
            "hint: train returned 0 but no metrics.json or per_column_metrics.csv in",
            str(out_dir),
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
