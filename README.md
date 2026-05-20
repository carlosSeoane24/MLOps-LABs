# MLOps-LABs

---

## LAB 2: Predicción de Fallos en Productos (Clasificación Binaria)

En esta segunda práctica hemos abordado un problema de **Machine Learning de clasificación binaria**. El objetivo principal es predecir la probabilidad de que un producto experimente un fallo (`failure` = 1) bajo cargas simuladas en un entorno del mundo real, basándonos en sus atributos fijos y los resultados de diversas pruebas de laboratorio.

### 🛠️ Flujo de trabajo y técnicas aplicadas:
1. **Limpieza y Preprocesamiento de Datos:**
   * **Imputación de nulos:** Se ha utilizado la mediana para rellenar los datos faltantes en las mediciones de los sensores numéricos (`SimpleImputer`).
   * **Codificación de variables categóricas:** Transformación de atributos de texto (ej. códigos de material) a valores numéricos mediante `LabelEncoder`.
2. **Modelado:**
   * Entrenamiento de un modelo robusto basado en árboles de decisión: **Random Forest Classifier**.
   * División del conjunto de datos (Train/Validation) para comprobar el rendimiento localmente.
3. **Evaluación:**
   * Uso de la métrica **ROC-AUC** para evaluar el rendimiento predictivo del modelo.
4. **Entregable:**
   * Generación del archivo final `submission.csv` con el ID del producto del conjunto de *test* y su probabilidad estimada de fallo.