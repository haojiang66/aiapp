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
            st.subheader("本次检测到的类别")
            cls_ids = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            lines = []
            for i, (cid, c) in enumerate(zip(cls_ids, confs)):
                lines.append(f"{i + 1}. {names[cid]} — {c:.2f}")
            st.text("\n".join(lines))
        else:
            st.info("未检测到目标，可调低置信度阈值或换一张图。")


if __name__ == "__main__":
    main()
