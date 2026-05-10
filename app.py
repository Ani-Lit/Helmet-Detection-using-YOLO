import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Helmet Detection AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: #0f172a;
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.15rem;
    margin-bottom: 2rem;
}

.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.metric-card {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.metric-number {
    font-size: 2rem;
    font-weight: 700;
    color: white;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.95rem;
}

img {
    border-radius: 18px;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================
st.markdown("""
<div class="main-title">
🪖 Helmet Detection AI
</div>

<div class="subtitle">
Helmet and head detection for industrial safety monitoring
</div>
""", unsafe_allow_html=True)

# =====================================
# MODEL LOADING
# =====================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:

    st.markdown("## ⚙️ Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.50,
        0.05
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        0.0,
        1.0,
        0.45,
        0.05
    )

    st.markdown("---")

    st.markdown("## 📊 Model Statistics")

    st.markdown("""
<div class="glass">

### YOLOv8m

- mAP50: **95.8%**
- Optimized for industrial environments
- Classes:
    - Helmet
    - Head

</div>
""", unsafe_allow_html=True)

# =====================================
# STATUS BAR
# =====================================
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">95.8%</div>
        <div class="metric-label">mAP50 Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">YOLOv8m</div>
        <div class="metric-label">Model Architecture</div>
    </div>
    """, unsafe_allow_html=True)

with colC:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">2</div>
        <div class="metric-label">Detection Classes</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================
# FILE UPLOADER
# =====================================
st.markdown("## 📤 Upload Image")

uploaded_image = st.file_uploader(
    "Drop an image below",
    type=["jpg", "jpeg", "png", "bmp"]
)

# =====================================
# INFERENCE
# =====================================
if uploaded_image is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_image.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Running AI Detection..."):

        results = model(
            image_rgb,
            conf=conf_threshold,
            iou=iou_threshold
        )

    annotated_image = results[0].plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    st.markdown("## 📷 Detection Results")

    left, right = st.columns(2)

    with left:
        st.markdown("### Original Image")
        st.image(image_rgb, use_column_width=True)

    with right:
        st.markdown("### AI Detection")
        st.image(annotated_image, use_column_width=True)

    detections = results[0].boxes

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================
    # RESULTS CARDS
    # =====================================
    total_helmets = 0
    total_heads = 0

    for box in detections:

        cls_id = int(box.cls)
        cls_name = results[0].names[cls_id]

        if cls_name.lower() == "helmet":
            total_helmets += 1
        else:
            total_heads += 1

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(detections)}</div>
            <div class="metric-label">Total Detections</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_helmets}</div>
            <div class="metric-label">Helmets</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_heads}</div>
            <div class="metric-label">No Helmet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================
    # DETAILED TABLE
    # =====================================
    if len(detections) > 0:

        st.markdown("## 📋 Detection Table")

        detection_data = []

        for i, box in enumerate(detections):

            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]

            if cls_name.lower() == "head":
                cls_name = "No Helmet"

            conf = box.conf.item()

            detection_data.append({
                "ID": i + 1,
                "Class": cls_name,
                "Confidence": f"{conf:.2f}"
            })

        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("⚠️ No objects detected")

# =====================================
# FOOTER
# =====================================
st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8;'>

Built using Streamlit and YOLOv8

</div>
""", unsafe_allow_html=True)
