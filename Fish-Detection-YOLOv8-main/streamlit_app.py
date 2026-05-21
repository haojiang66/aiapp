"""
水生生物检测应用：鱼类 / 底栖动物切换；支持上传文件或剪贴板图片。

运行（在 Fish-Detection-YOLOv8-main 目录下）:
  pip install -r requirements-app.txt
  streamlit run streamlit_app.py
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageGrab
from ultralytics import YOLO

from fish_info import FISH_INFO, get_fish_info

APP_ROOT = Path(__file__).resolve().parent
MACRO_ROOT = APP_ROOT.parent / "Macroinvertebrate-Detection-YOLOv8-main"

MACRO_INFO = {
    "bugs_and_beetles": {
        "zh_name": "蝽或甲虫",
        "intro": "礁区或淡水底质常见，许多种类为捕食性，是溪流生物评估中的敏感类群之一。",
    },
    "caddisflies": {
        "zh_name": "石蛾幼虫（石蚕）",
        "intro": "常具护壳，多出现在清洁、含氧较好的水流中，水质指示价值高。",
    },
    "crabs_and_shrimps": {
        "zh_name": "蟹与虾",
        "intro": "底栖或近底活动，对污染有一定耐受性，种类多样。",
    },
    "damselflies": {
        "zh_name": "豆娘",
        "intro": "幼虫水生，成虫陆生，常出现在静水或缓流环境。",
    },
    "dragonflies": {
        "zh_name": "蜻蜓",
        "intro": "幼虫（水虿）栖息水底，成体飞行能力强，幼虫为掠食性。",
    },
    "flat_worms": {
        "zh_name": "扁形虫",
        "intro": "身体扁平，多栖息在石块或沉木下，对水质变化较敏感。",
    },
    "leeches": {
        "zh_name": "水蛭",
        "intro": "常见于富含有机质的缓流或静水区，部分种类对污染耐受较强。",
    },
    "minnow_mayflies": {
        "zh_name": "小蜉蝣",
        "intro": "蜉蝣目幼虫，多指示清洁、高含氧水体，是 miniSASS 重要类群。",
    },
    "other_mayflies": {
        "zh_name": "其他蜉蝣",
        "intro": "除小蜉蝣外的蜉蝣类幼虫，同样多见于水质较好的溪流。",
    },
    "snails_clams_mussels": {
        "zh_name": "螺、蚌、贻贝",
        "intro": "软体动物类群，滤食或刮食藻类，分布与流速、底质密切相关。",
    },
    "stoneflies": {
        "zh_name": "石蛾",
        "intro": "石蛾目幼虫，喜冷水、高氧环境，是水质优良的重要指示生物。",
    },
    "true_flies": {
        "zh_name": "真蝇",
        "intro": "包括蚊类幼虫等，部分种类在有机污染较重的环境中数量增多。",
    },
    "worms": {
        "zh_name": "蠕虫",
        "intro": "环节动物等蠕虫类底栖生物，在泥沙底质中十分常见。",
    },
}


def get_macro_info(class_name: str) -> dict:
    info = MACRO_INFO.get(class_name)
    if info:
        return info
    return {
        "zh_name": "未知类别",
        "intro": f"暂无「{class_name}」的中文介绍。",
    }


SPECIES_OPTIONS = {
    "鱼类": {
        "weights": APP_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        "page_title": "鱼类检测",
        "title": "鱼类检测（YOLOv8）",
        "caption": "上传图片，识别珊瑚礁等相关鱼类类别。",
        "catalog_title": "全部可识别鱼类（中文名与简介）",
        "catalog": FISH_INFO,
        "get_info": get_fish_info,
    },
    "底栖动物": {
        "weights": MACRO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        "page_title": "底栖动物检测",
        "title": "底栖动物检测（YOLOv8）",
        "caption": "上传图片，识别 miniSASS 相关底栖无脊椎动物类别。",
        "catalog_title": "全部可识别底栖动物（中文名与简介）",
        "catalog": MACRO_INFO,
        "get_info": get_macro_info,
    },
}


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def read_system_clipboard() -> Image.Image | None:
    """读取本机系统剪贴板中的图片（Windows 截图 / 复制图片后有效）。"""
    data = ImageGrab.grabclipboard()
    if data is None:
        return None
    if isinstance(data, list):
        return Image.open(data[0]).convert("RGB")
    return data.convert("RGB")


def decode_data_url(data_url: str) -> Image.Image:
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def browser_paste_widget() -> str | None:
    """浏览器内粘贴区域：聚焦后 Ctrl+V，将图片以 data URL 回传。"""
    return components.html(
        """
        <div id="paste-area" tabindex="0"
             style="border:2px dashed #6c757d;padding:28px;text-align:center;
                    border-radius:8px;outline:none;cursor:text;background:#fafafa;">
          点击此处聚焦，然后按 <b>Ctrl+V</b> 粘贴图片
        </div>
        <script>
        const area = document.getElementById('paste-area');
        area.addEventListener('paste', (event) => {
          const items = event.clipboardData ? event.clipboardData.items : [];
          for (const item of items) {
            if (item.type && item.type.startsWith('image/')) {
              const file = item.getAsFile();
              const reader = new FileReader();
              reader.onload = (e) => {
                window.parent.postMessage(
                  {type: 'streamlit:setComponentValue', value: e.target.result},
                  '*'
                );
              };
              reader.readAsDataURL(file);
              event.preventDefault();
              area.innerHTML = '<b style="color:#198754">已粘贴图片</b>，稍候自动识别…';
              break;
            }
          }
        });
        </script>
        """,
        height=130,
    )


def load_input_image(input_mode: str) -> tuple[Image.Image | None, str]:
    """根据输入方式返回 RGB 图片与来源说明。"""
    if input_mode == "上传文件":
        uploaded = st.file_uploader(
            "选择图片",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="file_uploader",
        )
        if uploaded is not None:
            return Image.open(uploaded).convert("RGB"), "上传文件"
        return None, ""

    st.caption("先复制或截图（Win+Shift+S），再任选一种方式读取。")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("读取系统剪贴板", type="primary", use_container_width=True):
            img = read_system_clipboard()
            if img is None:
                st.warning("剪贴板中没有图片。请先截图或复制一张图。")
            else:
                st.session_state["clipboard_image"] = img
                st.success("已从系统剪贴板读取。")
    with col_b:
        if st.button("清除剪贴板图片", use_container_width=True):
            st.session_state.pop("clipboard_image", None)

    pasted = browser_paste_widget()
    if isinstance(pasted, str) and pasted.startswith("data:image"):
        st.session_state["clipboard_image"] = decode_data_url(pasted)

    img = st.session_state.get("clipboard_image")
    if img is not None:
        st.image(img, caption="当前剪贴板图片", use_container_width=True)
        return img, "剪贴板"
    return None, ""


def show_detections(results, get_info, names):
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("检测结果")
        cls_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()

        for i, (cid, score) in enumerate(zip(cls_ids, confs)):
            en_name = names[cid]
            info = get_info(en_name)
            st.markdown(
                f"**{i + 1}. {info['zh_name']}**  \n"
                f"模型类别：`{en_name}` · 置信度 **{score:.2f}**"
            )
            st.caption(info["intro"])
            if i < len(cls_ids) - 1:
                st.divider()

        seen = sorted({names[cid] for cid in cls_ids})
        if len(seen) > 1:
            with st.expander("本图涉及类别一览（去重）"):
                for en_name in seen:
                    info = get_info(en_name)
                    st.markdown(f"**{info['zh_name']}**（`{en_name}`）")
                    st.caption(info["intro"])
    else:
        st.info("未检测到目标，可调低置信度阈值或换一张图。")


def main():
    st.set_page_config(page_title="水生生物检测", layout="centered")

    st.sidebar.header("识别模式")
    species_key = st.sidebar.radio(
        "选择识别对象",
        list(SPECIES_OPTIONS.keys()),
        index=0,
    )
    cfg = SPECIES_OPTIONS[species_key]

    st.title(cfg["title"])
    st.caption(cfg["caption"])
    st.sidebar.caption(f"权重路径：\n`{cfg['weights']}`")

    if not cfg["weights"].is_file():
        st.error(
            f"未找到「{species_key}」模型权重：\n{cfg['weights']}\n\n"
            f"鱼类：在 Fish-Detection-YOLOv8-main 下完成训练。\n"
            f"底栖动物：在 Macroinvertebrate-Detection-YOLOv8-main 下完成 train.py。"
        )
        return

    conf = st.sidebar.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)

    st.subheader("输入图片")
    input_mode = st.radio(
        "图片来源",
        ["上传文件", "剪贴板"],
        horizontal=True,
        help="剪贴板：本机运行 Streamlit 时可用系统读取；也可在页面粘贴区 Ctrl+V。",
    )

    model = load_model(str(cfg["weights"]))
    image, source = load_input_image(input_mode)

    if image is not None:
        arr = np.array(image)
        results = model(arr, conf=conf, verbose=False)
        plotted = results[0].plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        st.image(
            plotted_rgb,
            caption=f"检测结果 · {species_key} · {source}",
            use_container_width=True,
        )
        show_detections(results, cfg["get_info"], results[0].names)

    with st.expander(cfg["catalog_title"]):
        for en_name, info in cfg["catalog"].items():
            st.markdown(f"**{info['zh_name']}** · `{en_name}`")
            st.caption(info["intro"])


if __name__ == "__main__":
    main()
