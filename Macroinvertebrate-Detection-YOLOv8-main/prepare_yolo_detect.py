"""
将 dataload.py 生成的按类别分文件夹数据，转换为 YOLOv8 检测格式。
每张图生成整图边界框（单目标分类图转检测，与鱼类项目相同的 detect 任务）。

输入: data/train/images/<类别>/*.jpg
输出: yolo_detect/train/images + train/labels, valid/...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# 单张图内仅一个目标：整图框 (class cx cy w h)，归一化坐标
FULL_IMAGE_BBOX = "0.5 0.5 1.0 1.0"


def load_class_names(data_root: Path) -> list[str]:
    classes_file = data_root / "metadata" / "classes.json"
    if classes_file.is_file():
        items = json.loads(classes_file.read_text(encoding="utf-8"))
        items = sorted(items, key=lambda x: x["id"])
        return [x["name_en"] for x in items]
    # 回退：按文件夹名排序（需与 data.yaml 一致）
    train_images = data_root / "train" / "images"
    return sorted(d.name for d in train_images.iterdir() if d.is_dir())


def class_to_id(class_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}


def write_data_yaml(yolo_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {yolo_root.resolve()}",
        "train: train/images",
        "val: valid/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    (yolo_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_split(
    src_images: Path,
    dst_images: Path,
    dst_labels: Path,
    split_prefix: str,
    name_to_id: dict[str, int],
) -> int:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for class_dir in sorted(src_images.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in name_to_id:
            raise ValueError(f"未知类别文件夹: {class_name}")
        cid = name_to_id[class_name]
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            stem = f"{split_prefix}_{cid:02d}_{count:05d}"
            out_img = dst_images / f"{stem}{img_path.suffix.lower()}"
            out_lbl = dst_labels / f"{stem}.txt"
            shutil.copy2(img_path, out_img)
            out_lbl.write_text(f"{cid} {FULL_IMAGE_BBOX}\n", encoding="utf-8")
            count += 1
    return count


def prepare(data_dir: Path, yolo_dir: Path, force: bool = False) -> Path:
    data_dir = data_dir.resolve()
    yolo_dir = yolo_dir.resolve()
    train_src = data_dir / "train" / "images"
    valid_src = data_dir / "valid" / "images"

    if not train_src.is_dir():
        raise FileNotFoundError(
            f"未找到 {train_src}，请先运行: python dataload.py"
        )

    marker = yolo_dir / ".prepared"
    if marker.is_file() and not force:
        print(f"已存在 YOLO 检测数据，跳过转换: {yolo_dir}")
        return yolo_dir / "data.yaml"

    if yolo_dir.exists() and force:
        shutil.rmtree(yolo_dir)

    class_names = load_class_names(data_dir)
    name_to_id = class_to_id(class_names)

    n_train = convert_split(
        train_src,
        yolo_dir / "train" / "images",
        yolo_dir / "train" / "labels",
        "tr",
        name_to_id,
    )
    n_valid = 0
    if valid_src.is_dir() and any(valid_src.iterdir()):
        n_valid = convert_split(
            valid_src,
            yolo_dir / "valid" / "images",
            yolo_dir / "valid" / "labels",
            "va",
            name_to_id,
        )

    yaml_path = yolo_dir / "data.yaml"
    write_data_yaml(yolo_dir, class_names)
    marker.write_text(f"train={n_train}\nvalid={n_valid}\n", encoding="utf-8")

    print(f"YOLO 检测数据已就绪: {yolo_dir}")
    print(f"  训练: {n_train} 张, 验证: {n_valid} 张, 类别: {len(class_names)}")
    return yaml_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--yolo-dir", type=Path, default=Path(__file__).parent / "yolo_detect")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    prepare(args.data_dir, args.yolo_dir, force=args.force)
