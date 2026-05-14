import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_epochs(log_path):
    epochs, tr_mse, tr_mae, val_mae = [], [], [], []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("epoch "):
                continue
            m = re.match(
                r"^epoch\s+(\d+)\s+"
                r"train_mse(?:_seq)?\s+([0-9.eE+-]+)\s+"
                r"train_mae_norm(?:_macro|_seq)?\s+([0-9.eE+-]+)\s+"
                r"val_mae_norm(?:_macro|_seq)?\s+([0-9.eE+-]+)",
                line.strip(),
            )
            if not m:
                continue
            epochs.append(int(m.group(1)))
            tr_mse.append(float(m.group(2)))
            tr_mae.append(float(m.group(3)))
            val_mae.append(float(m.group(4)))
    return epochs, tr_mse, tr_mae, val_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("rubert_runs"),
        help="каталог с подпапками run_*",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "coursework-latex/lyrata-kurs/images/training_stdout_mse_mae_curves.png"
        ),
    )
    ap.add_argument(
        "pairs",
        nargs="+",
        metavar="RUN:LABEL",
        help="например run_002_lstm:LSTM(1)",
    )
    ns = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = plt.cm.tab10.colors
    for i, raw in enumerate(ns.pairs):
        run_id, _, label = raw.partition(":")
        if not label:
            label = run_id
        log = ns.runs_dir / run_id / "stdout.log"
        ep, tmse, tmae, vmae = parse_epochs(log)
        if not ep:
            print("warn: no epochs in", log)
            continue
        c = colors[i % len(colors)]
        axes[0].plot(ep, tmse, "-", color=c, linewidth=1.4, label=label)
        axes[1].plot(ep, tmae, "--", color=c, linewidth=1.1, alpha=0.75)
        axes[1].plot(ep, vmae, "-", color=c, linewidth=1.35, label=label)
    axes[0].set_ylabel("train MSE (по строке epoch в логе)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("MAE norm (train --, val —)")
    axes[1].set_xlabel("эпоха")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Динамика обучения по stdout.log нескольких прогонов", fontsize=11)
    fig.tight_layout()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ns.out, dpi=160)
    print("saved", ns.out)


if __name__ == "__main__":
    main()
