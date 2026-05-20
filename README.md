# MLOps-LABs

# 🚀 LAB 6: Capstone Project - Docker Containerized MLOps Architecture

[![Docker](https://img.shields.io/badge/Container-Docker-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/)

Este proyecto representa el **Capstone Project** de la asignatura de MLOps. El objetivo ha sido empaquetar toda la solución de inferencia (modelo + API + interfaz) en un entorno aislado, reproducible y escalable utilizando **Docker Compose**.

## 🏗️ Arquitectura del Sistema

La solución se ha desplegado bajo un enfoque de **microservicios**, separando la lógica de negocio (Backend) de la capa de presentación (Frontend):



### Componentes:
1.  **Backend (FastAPI):** Servicio de inferencia que descarga dinámicamente el modelo desde Hugging Face Hub al iniciar. Implementa un `HEALTHCHECK` que asegura que el modelo esté cargado antes de que el frontend intente comunicarse con él.
2.  **Frontend (Streamlit):** Interfaz web que permite al usuario interactuar con el modelo. Se comunica con el backend a través de la red privada de Docker, garantizando seguridad y baja latencia.
3.  **Orquestación:** Configurado mediante `docker-compose.yml`, gestionando la creación de redes internas, el mapeo de puertos y la inyección segura de variables de entorno (tokens).

## 🚀 Guía de Despliegue Rápido

Para desplegar esta arquitectura en cualquier entorno, asegúrate de tener Docker instalado y sigue estos pasos:

1. **Configurar el Token de Hugging Face:**
   ```bash
   export HF_TOKEN="tu_token_aqui"