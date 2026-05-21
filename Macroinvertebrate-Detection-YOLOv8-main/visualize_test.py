"""
在验证集上可视化底栖生物检测效果，并生成指标图与样本对比图。

用法:
  python visualize_test.py
  python visualize_test.py --num-samples 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
from ultralytics import YOLO

from macro_info import load_class_zh, zh_name

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
DATA_YAML = ROOT / "yolo_detect" / "data.yaml"
VAL_IMAGES = ROOT / "yolo_detect" / "valid" / "images"
OUT_DIR = ROOT / "runs" / "detect" / "test_vis"


def run_validation(model: YOLO) -> None:
    print("\n=== 验证集指标 ===")
    metrics = model.val(data=str(DATA_YAML), split="val", plots=True, verbose=False)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"  曲线图已更新至: {ROOT / 'runs' / 'detect' / 'val'}")


def plot_sample_grid(model: YOLO, num_samples: int, seed: int, conf: float) -> Path:
    images = sorted(VAL_IMAGES.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"未找到验证图片: {VAL_IMAGES}")

    rng = random.Random(seed)
    picks = rng.sample(images, min(num_samples, len(images)))
    class_zh = load_class_zh()
    names = model.names

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, img_path in zip(axes, picks):
        results = model.predict(str(img_path), conf=conf, verbose=False)
        plotted = results[0].plot()
        plotted_rgb = plotted[:, :, ::-1]
        ax.imshow(plotted_rgb)
        ax.axis("off")
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            cid = int(boxes.cls[0].item())
            score = float(boxes.conf[0].item())
            title = f"{zh_name(cid, names)}\n{score:.2f}"
        else:
            title = "未检测到"
        ax.set_title(title, fontsize=9)

    for ax in axes[len(picks) :]:
        ax.axis("off")

    fig.suptitle("底栖生物检测 — 验证集样本（YOLOv8）", fontsize=14, y=1.01)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "val_samples_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n样本对比图: {out_path}")
    return out_path


def copy_training_plots() -> None:
    train_dir = ROOT / "runs" / "detect" / "train"
    plots = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in plots:
        src = train_dir / name
        if src.is_file():
            dst = OUT_DIR / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                dst.write_bytes(src.read_bytes())
    print(f"训练/验证图表副本: {OUT_DIR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=16, help="网格展示的样本数")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-val", action="store_true", help="跳过完整 val（仅画样本图）")
    args = p.parse_args()

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"未找到权重: {WEIGHTS}，请先完成 train.py")

    print(f"加载模型: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    copy_training_plots()
    if not args.skip_val:
        run_validation(model)
    plot_sample_grid(model, args.num_samples, args.seed, args.conf)

    print("\n打开以下目录查看全部可视化结果:")
    print(f"  {OUT_DIR}")
    print(f"  {ROOT / 'runs' / 'detect' / 'train'}")


if __name__ == "__main__":
    main()
