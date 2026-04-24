import streamlit as st
import os
from ultralytics import YOLO

# Configuramos la página
st.set_page_config(page_title="Detector PPE", layout="wide")

@st.cache_resource
def load_model():
    # Verificamos si el archivo existe y tiene tamaño real
    model_path = "best.pt"
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return YOLO(model_path)
    else:
        st.error(f"Error: El archivo {model_path} no se encuentra o está corrupto. Tamaño: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")
        return None

st.title("Detector de EPP (PPE)")
model = load_model()

if model:
    st.success("Modelo cargado correctamente.")
    # Aquí iría tu lógica para subir imágenes y detectar
else:
    st.warning("No se pudo cargar el modelo. Verifica que best.pt esté en la raíz.")
