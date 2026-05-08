Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Buckets
new
Docs
Pricing


Hugging Face is way more fun with friends and colleagues! 🤗 Join an organization
Spaces:
AniHug
/
Helmet_Detection


like
0

App
Files
Community
Settings
Helmet_Detection
/
app.py

AniHug's picture
AniHug
Rename app.py.txt to app.py
e55d535
verified
about 10 hours ago
raw

Copy download link
history
blame
edit
delete
3.23 kB
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Page config
st.set_page_config(page_title="Safety Helmet Detector", layout="wide")
st.title("🛡️ Safety Helmet & Head Detector")

st.markdown("""
Detects **helmets** and **bare heads** in images using YOLOv8m.

**Model Performance:**
- mAP50: 0.968
- Trained on: Construction/industrial safety scenarios (bare heads + hard helmets)
- Classes: Head, Helmet
""")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Sidebar settings
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, 0.05)

# Upload image
st.subheader("📤 Upload Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    with st.spinner("🔍 Detecting..."):
        results = model(image_rgb, conf=conf_threshold, iou=iou_threshold)
    
    # Draw results
    annotated_image = results[0].plot()[:, :, ::-1]  # BGR to RGB
    
    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original", use_column_width=True)
    with col2:
        st.image(annotated_image, caption="Detections", use_column_width=True)
    
    # Show statistics
    st.subheader("📊 Results")
    detections = results[0].boxes
    
    if len(detections) > 0:
        class_counts = {}
        for box in detections:
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Objects Detected:**")
            for cls_name, count in class_counts.items():
                st.write(f"- {cls_name}: {count}")
        
        with col2:
            st.write(f"**Total:** {len(detections)}")
        
        # Detailed table
        detection_data = []
        for i, box in enumerate(detections):
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            conf = box.conf.item()
            detection_data.append({
                "ID": i + 1,
                "Class": cls_name,
                "Confidence": f"{conf:.3f}",
            })
        st.dataframe(detection_data, use_container_width=True)
    else:
        st.warning("⚠️ No objects detected.")

st.markdown("---")
st.markdown("""
**Model Info:**
- Architecture: YOLOv8m
- mAP50: 0.968
- Training data: Bare heads + hard helmets (construction/industrial)

Made with ❤️ using Streamlit + YOLOv8
""")
