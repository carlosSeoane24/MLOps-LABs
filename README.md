# MLOps-LABs

---

## 🚀 LAB 5: Cloud Deployment en Hugging Face Ecosystem

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CarlosSeoane/iris-space)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4)](https://gradio.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

En este laboratorio hemos dado el salto a la nube explorando el ecosistema de **Hugging Face**, utilizándolo como nuestro registro centralizado de MLOps. Hemos completado el ciclo de vida completo de un modelo de Machine Learning de forma remota, desde la ingesta de datos hasta el despliegue en producción con una interfaz gráfica.

### 🏗️ Arquitectura y Flujo de Trabajo

El proyecto se divide en tres componentes clave interactuando entre sí:

1. 📊 **Gestión de Datos (Data Hub):** Transformación del dataset local *Iris* al formato nativo de la librería `datasets` y despliegue en el Hub para su versionado y acceso remoto.
2. 🧠 **Registro de Modelos (Model Hub):** Entrenamiento de un clasificador `DecisionTreeClassifier` con Scikit-Learn. El artefacto final (`.joblib`) se serializa y se sube automáticamente al registro de modelos de Hugging Face mediante la API de `huggingface_hub`.
3. 💻 **Despliegue y Servido (Spaces):** Creación de una aplicación web interactiva utilizando el framework **Gradio**. La aplicación es agnóstica a los archivos locales: descarga automáticamente el modelo desde el registro remoto en tiempo real y expone una interfaz amigable para el usuario final.

### 🔗 Enlaces a los Artefactos Desplegados

A continuación se encuentran los enlaces directos a los recursos alojados en los servidores de Hugging Face:

* **📦 Dataset Original:** [CarlosSeoane/iris](https://huggingface.co/datasets/CarlosSeoane/iris)
* **🤖 Modelo Entrenado (Decision Tree):** [CarlosSeoane/iris-dt](https://huggingface.co/CarlosSeoane/iris-dt)
* **🌐 Web App en Producción (Gradio Space):** [CarlosSeoane/iris-space](https://huggingface.co/spaces/CarlosSeoane/iris-space)

### 🛠️ Tecnologías Utilizadas
* **Scripts de subida:** `huggingface-cli`, `huggingface_hub`
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
* **Interfaz de Usuario:** `gradio`