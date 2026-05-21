"""
从 Hugging Face 下载 CGIAR 水生无脊椎动物图像数据集到本地。

数据集: CGIAR/aquatic-macroinvertebrate-images
约 1300 张图、13 类（miniSASS 分组），约 3.1 GB。

用法（在本目录下）:
  pip install -r requirements-dataload.txt
  python dataload.py
  python dataload.py --output-dir ./data --val-ratio 0.2

输出目录结构:
  data/
    train/images/<类别名>/*.jpg
    valid/images/<类别名>/*.jpg
    metadata/classes.json
    metadata/manifest.csv
    data.yaml          # 供 YOLO 分类/后续训练参考（非检测框标注）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "CGIAR/aquatic-macroinvertebrate-images"

# Hugging Face 官方 ClassLabel.names（label 下标与此列表顺序一致）
ZH_BY_CLASS_EN = {
    "bugs_and_beetles": "蝽或甲虫",
    "caddisflies": "石蛾幼虫（石蚕）",
    "crabs_and_shrimps": "蟹与虾",
    "damselflies": "豆娘",
    "dragonflies": "蜻蜓",
    "flat_worms": "扁形虫",
    "leeches": "水蛭",
    "minnow_mayflies": "小蜉蝣",
    "other_mayflies": "其他蜉蝣",
    "snails_clams_mussels": "螺、蚌、贻贝",
    "stoneflies": "石蛾",
    "true_flies": "真蝇",
    "worms": "蠕虫",
}


def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip().lower()).strip("_")


def resolve_class_names(label_feature) -> list[str]:
    """优先使用数据集自带的 ClassLabel.names。"""
    if hasattr(label_feature, "names") and label_feature.names:
        return [slugify(n) for n in label_feature.names]
    return list(ZH_BY_CLASS_EN.keys())


def class_name_zh(class_en: str) -> str:
    return ZH_BY_CLASS_EN.get(class_en, "")


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    meta = root / "metadata"
    train_img = root / "train" / "images"
    valid_img = root / "valid" / "images"
    meta.mkdir(parents=True, exist_ok=True)
    train_img.mkdir(parents=True, exist_ok=True)
    valid_img.mkdir(parents=True, exist_ok=True)
    return train_img, valid_img, meta


def write_classes_json(meta_dir: Path, class_names: list[str]) -> None:
    classes = [
        {
            "id": i,
            "name_en": class_names[i],
            "name_zh": class_name_zh(class_names[i]),
        }
        for i in range(len(class_names))
    ]
    (meta_dir / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_data_yaml(root: Path, class_names: list[str]) -> None:
    """生成 YOLO 分类任务可用的 data.yaml（非目标检测框）。"""
    lines = [
        f"path: {root.resolve()}",
        "train: train/images",
        "val: valid/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        zh = class_name_zh(name)
        lines.append(f"  {i}: {name}  # {zh}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=92)


def download_dataset(
    output_dir: Path,
    cache_dir: str | None,
    val_ratio: float,
    seed: int,
    max_samples: int | None,
    resume: bool,
) -> None:
    output_dir = output_dir.resolve()
    train_img, valid_img, meta_dir = ensure_dirs(output_dir)

    load_kwargs: dict = {"split": "train"}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    print(f"正在从 Hugging Face 加载: {DATASET_ID}")
    print(f"保存到: {output_dir}")
    print("数据量约 3.1 GB，首次下载可能需要较长时间，请保持网络畅通。\n")

    ds = load_dataset(DATASET_ID, **load_kwargs)
    split = ds["train"] if isinstance(ds, dict) else ds
    class_names = resolve_class_names(split.features["label"])
    num_classes = len(class_names)

    if num_classes != 13:
        print(f"提示: 数据集报告 {num_classes} 类，将按实际 label 映射保存。")

    write_classes_json(meta_dir, class_names)
    write_data_yaml(output_dir, class_names)

    manifest_path = meta_dir / "manifest.csv"
    manifest_mode = "a" if resume and manifest_path.exists() else "w"
    existing = set()
    if resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["filepath"])

    rng = random.Random(seed)
    counts = {"train": 0, "valid": 0, "skipped": 0}

    with manifest_path.open(manifest_mode, encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["filepath", "split", "label_id", "class_en", "class_zh"],
        )
        if manifest_mode == "w":
            writer.writeheader()

        total = len(split) if max_samples is None else min(max_samples, len(split))
        for idx, sample in enumerate(
            tqdm(split, total=total, desc="下载并保存", unit="张")
        ):
            if max_samples is not None and idx >= max_samples:
                break

            label_id = int(sample["label"])
            if label_id >= len(class_names):
                class_en = f"unknown_{label_id}"
                class_zh = ""
            else:
                class_en = class_names[label_id]
                class_zh = class_name_zh(class_en)

            use_val = rng.random() < val_ratio
            split_name = "valid" if use_val else "train"
            base = valid_img if use_val else train_img
            rel = Path(split_name) / "images" / class_en / f"{idx:05d}.jpg"
            out_path = output_dir / rel

            if resume and (str(rel).replace("\\", "/") in existing or out_path.is_file()):
                counts["skipped"] += 1
                continue

            save_image(sample["image"], out_path)
            writer.writerow(
                {
                    "filepath": str(rel).replace("\\", "/"),
                    "split": split_name,
                    "label_id": label_id,
                    "class_en": class_en,
                    "class_zh": class_zh,
                }
            )
            counts[split_name] += 1

    print("\n完成。")
    print(f"  训练集: {counts['train']} 张 -> {train_img}")
    print(f"  验证集: {counts['valid']} 张 -> {valid_img}")
    if counts["skipped"]:
        print(f"  跳过(已存在): {counts['skipped']} 张")
    print(f"  类别表: {meta_dir / 'classes.json'}")
    print(f"  清单:   {manifest_path}")
    print(f"  配置:   {output_dir / 'data.yaml'}")


def parse_args():
    p = argparse.ArgumentParser(description="下载水生无脊椎动物数据集到本地")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="本地保存根目录（默认 ./data）",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Hugging Face 缓存目录（可选）",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="划入验证集的比例（默认 0.2）",
    )
    p.add_argument("--seed", type=int, default=42, help="划分 train/val 的随机种子")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅下载前 N 张（调试用）",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="不跳过已下载文件，全部重新保存",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        resume=not args.no_resume,
    )
