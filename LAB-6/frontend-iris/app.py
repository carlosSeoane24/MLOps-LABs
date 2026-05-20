import streamlit as st
import requests

st.title("🌸 Clasificador de Flores Iris")
st.write("Conectado al Backend mediante Docker Compose")

sepal_l = st.slider("Longitud Sépalo", 4.0, 8.0, 5.1)
sepal_w = st.slider("Anchura Sépalo", 2.0, 5.0, 3.5)
petal_l = st.slider("Longitud Pétalo", 1.0, 7.0, 1.4)
petal_w = st.slider("Anchura Pétalo", 0.1, 3.0, 0.2)

if st.button("Hacer Predicción"):
    datos = {
        "sepal_length": sepal_l, 
        "sepal_width": sepal_w, 
        "petal_length": petal_l, 
        "petal_width": petal_w
    }
    # Llama al contenedor interno de Docker llamado "backend"
    res = requests.post("http://backend:8000/predict", json=datos)
    
    if res.status_code == 200:
        clase = res.json()["class_name"]
        st.success(f"La IA predice que es una: **{clase.upper()}**")
    else:
        st.error("El backend aún no está listo o hay un error.")