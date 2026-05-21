"""底栖生物类别中文名（仅前端/可视化，不参与训练）。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES_JSON = ROOT / "data" / "metadata" / "classes.json"

ZH_BY_EN = {
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


def load_class_zh() -> dict[int, str]:
    if CLASSES_JSON.is_file():
        items = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
        return {x["id"]: x.get("name_zh") or ZH_BY_EN.get(x["name_en"], x["name_en"]) for x in items}
    return {i: v for i, v in enumerate(ZH_BY_EN.values())}


def zh_name(class_id: int, names: dict) -> str:
    en = names.get(class_id, str(class_id))
    zh = ZH_BY_EN.get(en, "")
    return f"{zh} ({en})" if zh else en
