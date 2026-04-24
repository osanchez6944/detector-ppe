import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Detector EPP", layout="centered")

st.title("🦺 Detector de EPP (PPE)")

# Cargar modelo UNA sola vez
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
st.success("Modelo cargado correctamente.")

# -------------------------------
# OPCIÓN 1: SUBIR IMAGEN
# -------------------------------
st.subheader("📂 Subir imagen")
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    img_array = np.array(image)

    with st.spinner("Detectando..."):
        results = model(img_array)
        result_img = results[0].plot()

    st.image(result_img, caption="Resultado", use_column_width=True)

# -------------------------------
# OPCIÓN 2: CÁMARA
# -------------------------------
st.subheader("📸 Usar cámara")
camera_image = st.camera_input("Toma una foto")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Foto capturada", use_column_width=True)

    img_array = np.array(image)

    with st.spinner("Detectando..."):
        results = model(img_array)
        result_img = results[0].plot()

    st.image(result_img, caption="Resultado", use_column_width=True)
