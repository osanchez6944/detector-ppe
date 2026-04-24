import streamlit as st
from ultralytics import YOLO

# ESTO ES LO QUE HACE QUE LA APP CARGUE
@st.cache_resource
def load_model():
    # Asegúrate de que best.pt esté en la raíz del repo
    return YOLO("best.pt")

st.title("Detector PPE")
model = load_model() # Solo se carga una vez en memoria
st.write("Modelo cargado exitosamente.")
