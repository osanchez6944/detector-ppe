import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Cargar tu modelo entrenado (asegúrate de tener best.pt en la carpeta)
model = YOLO('best.pt')

st.title("Detector de EPP (Casco, Chaleco, Guantes)")

uploaded_file = st.file_uploader("Sube una foto...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Realizar detección
    results = model.predict(image)
    
    # Mostrar resultados
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Resultado de la detección')