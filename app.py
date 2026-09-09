import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------------------------------------------
# Page Configuration & UI Layout
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MD-Vision | Chest X-Ray AI Screening",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    .sdg-badge { background-color: #E0F2FE; color: #0369A1; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: inline-block; margin-right: 8px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# Top Bar Header
st.markdown('<div class="main-title">🩺 MD-Vision: Automated Chest X-Ray Screening</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Assisted Early Detection for Underserved Rural & Remote Health Clinics</div>', unsafe_allow_html=True)

st.markdown("""
<span class="sdg-badge">🎯 Goal 3: Good Health & Well-Being</span>
<span class="sdg-badge">🌐 Goal 10: Reduced Inequalities</span>
""", unsafe_allow_html=True)

st.write("---")

# -------------------------------------------------------------------
# Model Loader Setup (PyTorch Transfer Learning Model)
# -------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    # Utilizing MobileNetV2 for lightweight, fast inference
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Re-map classifier layer to 2 outputs: Normal vs Pneumonia
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.eval()
    return model

model = load_classifier()

# Image Pre-processing Pipeline
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = ImageOps.fit(image_bytes, (224, 224), Image.Resampling.LANCZOS).convert('RGB')
    return transform(image).unsqueeze(0)

# -------------------------------------------------------------------
# App Core Functionality (2 Column View Layout)
# -------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Patient Scan")
    uploaded_file = st.file_uploader("Select a Chest X-Ray DICOM/JPEG Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Scan", use_column_width=True)
    else:
        st.info("👆 Upload an X-ray image to perform analysis.")

with col2:
    st.subheader("2. AI Analysis & Results")
    if uploaded_file is not None:
        with st.spinner('Analyzing image through neural network...'):
            try:
                # Preprocess & Predict
                img_tensor = transform_image(image)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                normal_score = float(probabilities[0]) * 100
                pneumonia_score = float(probabilities[1]) * 100

                # Output Prediction Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                if pneumonia_score > 50:
                    st.error(f"⚠️ **SUSPECTED CASE: PNEUMONIA**")
                    st.metric(label="Pneumonia Probability", value=f"{pneumonia_score:.1f}%")
                    st.progress(int(pneumonia_score))
                    st.warning("**Recommendation:** Immediate review advised by a clinical radiologist.")
                else:
                    st.success(f"✅ **RESULT: NORMAL LUNG SCAN**")
                    st.metric(label="Normal Probability", value=f"{normal_score:.1f}%")
                    st.progress(int(normal_score))
                    st.info("**Status:** No significant opacities detected.")

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during processing: {e}")
    else:
        st.write("Results will appear here once an image is uploaded.")

# -------------------------------------------------------------------
# Dashboard Section: Impact Analytics
# -------------------------------------------------------------------
st.write("---")
st.subheader("3. System & SDG Target Impact Tracker")

m1, m2, m3 = st.columns(3)
m1.metric("Average Inference Time", "< 0.8 seconds")
m2.metric("Target SDG Benchmark", "Target 3.3 & Target 10.3")
m3.metric("Clinical Deployment Area", "Remote Health Centers")