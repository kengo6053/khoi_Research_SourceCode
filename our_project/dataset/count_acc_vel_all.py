#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_acc_vel_all.py
────────────────────
measurements/*.json から velocity / acceleration を収集し、
0.5 m/s 刻み（≤-2, −2~−1.5, …, 1.5~2, ≥2）の 10 区分で

  • PNG ヒストグラムを描画
  • 各区間のカウントをテキストで保存

Usage
-----
python count_acc_vel_all.py <root_folder> <dataset_list.txt>
"""

import os
import sys
import json
from typing import List, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────
# ビン定義
# ────────────────────────────────
BIN_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0,
                  0.5, 1.0, 1.5, 2.0]                # 9 境界 → 10 区分

BIN_LABELS = (
    ["≤-2"] +
    [f"{e:.1f}~{e+0.5:.1f}" for e in np.arange(-2.0, 2.0, 0.5)] +
    ["≥2"]
)

BIN_CENTERS = (
    [-2.25] +
    [t + 0.25 for t in BIN_THRESHOLDS[:-1]] +
    [2.25]
)

BAR_WIDTH = 0.45            # 区間幅 0.5 の 90%


# ────────────────────────────────
# ユーティリティ
# ────────────────────────────────
def load_dataset_list(path: str) -> Set[str]:
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def collect_data(root: str, targets: Set[str]) -> Tuple[List[float], List[float]]:
    vels, accs = [], []
    for sub in os.listdir(root):
        if targets and sub not in targets:
            continue
        mdir = os.path.join(root, sub, "measurements")
        if not os.path.isdir(mdir):
            continue
        for fn in os.listdir(mdir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(mdir, fn), encoding="utf-8") as fp:
                d = json.load(fp)
            vels.append(float(d.get("velocity", 0.0)))
            accs.append(float(d.get("acceleration", 0.0)))
    return vels, accs


def bin_count(data: List[float]) -> np.ndarray:
    """BIN_THRESHOLDS で 10 区分に分類し頻度を返す。"""
    idx = np.digitize(data, BIN_THRESHOLDS, right=True)   # 0〜9
    return np.bincount(idx, minlength=len(BIN_LABELS))


def save_histogram(counts: np.ndarray, title: str,
                   xlabel: str, ylabel: str, out_path: str) -> None:
    """counts を棒グラフとして保存。"""
    plt.figure()
    plt.bar(BIN_CENTERS, counts, width=BAR_WIDTH, align="center", alpha=0.8)

    plt.xticks(BIN_CENTERS, BIN_LABELS, rotation=45, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_counts_txt(counts: np.ndarray, out_path: str, header: str) -> None:
    """ラベルとカウントをタブ区切りテキストで保存。"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for label, cnt in zip(BIN_LABELS, counts):
            f.write(f"{label}\t{cnt}\n")


# ────────────────────────────────
# メイン
# ────────────────────────────────
def main() -> None:
    if len(sys.argv) < 3:
        print("Usage:\n  python count_acc_vel_all.py <root_folder> <dataset_list.txt>")
        sys.exit(1)

    root, list_file = sys.argv[1:3]
    targets = load_dataset_list(list_file)

    vels, accs = collect_data(root, targets)
    if not vels or not accs:
        print("No data found for the specified datasets.")
        sys.exit(1)

    # ── カウント集計 ───────────────────
    vel_counts = bin_count(vels)
    acc_counts = bin_count(accs)

    # ── 画像保存 ─────────────────────
    vel_png = os.path.join(root, "velocity_distribution.png")
    acc_png = os.path.join(root, "acceleration_distribution.png")
    save_histogram(vel_counts, "Combined Velocity Distribution",
                   "Velocity [m/s]", "Frequency", vel_png)
    save_histogram(acc_counts, "Combined Acceleration Distribution",
                   "Acceleration [m/s²]", "Frequency", acc_png)

    # ── テキスト保存 ──────────────────
    vel_txt = os.path.join(root, "velocity_counts.txt")
    acc_txt = os.path.join(root, "acceleration_counts.txt")
    write_counts_txt(vel_counts, vel_txt, "Velocity Counts")
    write_counts_txt(acc_counts, acc_txt, "Acceleration Counts")

    # ── 完了メッセージ ────────────────
    print("Outputs saved:")
    print(f"- {vel_png}")
    print(f"- {acc_png}")
    print(f"- {vel_txt}")
    print(f"- {acc_txt}")


if __name__ == "__main__":
    main()

