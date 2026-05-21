"""
底栖生物 YOLOv8 检测模型训练（流程对齐 Fish-Detection 项目，不修改 fish 目录）。

步骤:
  1. python dataload.py          # 若尚无 data/
  2. python prepare_yolo_detect.py
  3. python train.py

产出权重（与 fish 相同相对路径）:
  runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def default_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "0"
    return "cpu"
DATA_DIR = ROOT / "data"
YOLO_DIR = ROOT / "yolo_detect"
WEIGHTS_OUT = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"


def ensure_raw_data() -> None:
    train_images = DATA_DIR / "train" / "images"
    if train_images.is_dir() and any(train_images.iterdir()):
        return
    print("未找到 data/train/images，正在运行 dataload.py 下载数据集…")
    subprocess.run(
        [sys.executable, str(ROOT / "dataload.py")],
        cwd=str(ROOT),
        check=True,
    )


def ensure_yolo_detect(force_prepare: bool) -> Path:
    from prepare_yolo_detect import prepare

    yaml_path = YOLO_DIR / "data.yaml"
    if not yaml_path.is_file() or force_prepare:
        prepare(DATA_DIR, YOLO_DIR, force=force_prepare)
    return yaml_path


def train(
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    model_name: str,
    workers: int,
    force_prepare: bool,
) -> Path:
    data_yaml = ensure_yolo_detect(force_prepare)

    weights = ROOT / model_name
    if not weights.is_file():
        weights = model_name  # 由 Ultralytics 自动下载 yolov8n.pt

    print(f"\n开始训练 | 数据: {data_yaml}")
    print(f"预训练权重: {model_name} | epochs={epochs} batch={batch} imgsz={imgsz} device={device}\n")

    model = YOLO(str(weights))
    model.train(
        task="detect",
        mode="train",
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    if not WEIGHTS_OUT.is_file():
        raise FileNotFoundError(f"训练结束但未找到权重: {WEIGHTS_OUT}")

    print(f"\n训练完成，最佳权重: {WEIGHTS_OUT}")
    return WEIGHTS_OUT


def parse_args():
    p = argparse.ArgumentParser(description="训练底栖生物 YOLOv8 检测模型")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数（fish 项目默认 50）")
    p.add_argument("--batch", type=int, default=4, help="批大小（原图较大，12GB 显存建议 4–8）")
    p.add_argument("--imgsz", type=int, default=416, help="输入尺寸（与 fish 导出尺寸一致，省显存）")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda:0 / cpu；默认自动选 GPU（可用时）",
    )
    p.add_argument("--model", type=str, default="yolov8n.pt", help="预训练权重")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 线程数")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="不自动运行 dataload（data/ 必须已存在）",
    )
    p.add_argument(
        "--force-prepare",
        action="store_true",
        help="强制重新生成 yolo_detect/",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not (DATA_DIR / "train" / "images").is_dir():
        if args.skip_download:
            sys.exit("缺少 data/train/images，请先运行 python dataload.py")
        ensure_raw_data()

    device = default_device(args.device)
    if device == "cpu":
        print("警告: 当前 PyTorch 未检测到 CUDA，将使用 CPU。")
        print("若有 NVIDIA 显卡，请安装 GPU 版 PyTorch: https://pytorch.org/get-started/locally/\n")
    else:
        name = torch.cuda.get_device_name(int(device) if device.isdigit() else 0)
        print(f"使用 GPU: {name} (device={device})\n")

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        model_name=args.model,
        workers=args.workers,
        force_prepare=args.force_prepare,
    )
