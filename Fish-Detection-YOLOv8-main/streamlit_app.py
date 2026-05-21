"""
鱼类检测简易应用：浏览器内上传图片并查看检测框。
运行（在项目根目录 Fish-Detection-YOLOv8-main 下）:
  pip install -r requirements-app.txt
  streamlit run streamlit_app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from fish_info import FISH_INFO, get_fish_info

WEIGHTS = Path(__file__).resolve().parent / "runs" / "detect" / "train" / "weights" / "best.pt"


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def main():
    st.set_page_config(page_title="鱼类检测", layout="centered")
    st.title("鱼类检测（YOLOv8）")
    st.caption("上传一张图片，模型会标出检测到的鱼类相关类别。")

    if not WEIGHTS.is_file():
        st.error(
            f"未找到权重文件：{WEIGHTS}\n"
            "请先完成训练或把 `best.pt` 放到上述路径。"
        )
        return

    conf = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
    uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "webp", "bmp"])

    model = load_model(str(WEIGHTS))

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        arr = np.array(image)
        results = model(arr, conf=conf, verbose=False)
        plotted = results[0].plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        st.image(plotted_rgb, caption="检测结果", use_container_width=True)

        names = results[0].names
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.subheader("检测结果")
            cls_ids = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()

            for i, (cid, score) in enumerate(zip(cls_ids, confs)):
                en_name = names[cid]
                info = get_fish_info(en_name)
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
                        info = get_fish_info(en_name)
                        st.markdown(f"**{info['zh_name']}**（`{en_name}`）")
                        st.caption(info["intro"])
        else:
            st.info("未检测到目标，可调低置信度阈值或换一张图。")

    with st.expander("全部可识别鱼类（中文名与简介）"):
        for en_name, info in FISH_INFO.items():
            st.markdown(f"**{info['zh_name']}** · `{en_name}`")
            st.caption(info["intro"])


if __name__ == "__main__":
    main()
