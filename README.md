# MLOps-LABs

---

## LAB 4: MLflow y Predicción de Diabetes

### Entregable 1: Selección del Mejor Modelo
Se ha realizado un *Grid Search* evaluando diferentes parámetros de un modelo Random Forest. La métrica elegida para decidir el mejor modelo a desplegar ha sido el **Recall**. 
* **Justificación Médica:** En el diagnóstico de la diabetes, un Falso Negativo (decirle a un paciente enfermo que está sano) es extremadamente peligroso porque se queda sin tratamiento y su condición empeorará. Es preferible tener Falsos Positivos, ya que pruebas posteriores lo descartarían sin perjuicio grave. Por tanto, se priorizó el parámetro `class_weight='balanced'` para maximizar la detección de positivos reales.

### Entregables 2 y 3: Script de API y Conclusiones Finales
Se ha aislado un conjunto de Test puro para no contaminar el entrenamiento y evaluar la generalización del modelo. Al consultar la API de MLflow (puerto 5000) con este set usando un script externo (`evaluar_api.py`), hemos obtenido las siguientes métricas reales:
* **Accuracy:** 0.7013 (70.13%)
* **Precision:** 0.5312 (53.12%)
* **Recall:** 0.6800 (68.00%)
* **F1-Score:** 0.5965 (59.65%)

**Conclusión:**
Queda demostrado el inmenso valor de herramientas MLOps como MLflow. Hemos podido trazar experimentos paralelos, seleccionar el modelo que mejor se alinea con las restricciones del dominio de negocio (ámbito médico) e instantáneamente desplegarlo como una API REST funcional. Las métricas confirman que el modelo generaliza bien y el servidor responde correctamente a JSONs con nuevos pacientes.