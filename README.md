# MLOps-LABs

---

## LAB 3: Continuous Training (CT) y Quality Gates

En esta fase del proyecto hemos evolucionado el nivel de madurez MLOps para implementar **Continuous Training (Entrenamiento Continuo)**. El servidor ya no solo realiza inferencias sobre un modelo estático, sino que incorpora endpoints para recibir nuevos datos etiquetados desde producción y reentrenarse "en caliente" sin interrupción del servicio.

### 🛠️ Características implementadas:
* **Gestión del Historial:** Registro persistente de versiones (`training_history.json`), tanto de los modelos promovidos como de los entrenamientos fallidos para garantizar la auditabilidad y el diagnóstico de Data Drift.
* **Entrenamiento Incremental vs Desde Cero:** Capacidad de la API para decidir si anexar las nuevas muestras al conjunto de datos histórico (`accumulated_data.joblib`) o descartarlo en caso de cambios drásticos.
* **Políticas de Calidad Avanzadas (Quality Gates):** * `any_improvement`: Permite la activación del nuevo modelo si supera o iguala la métrica anterior (`>=`).
    * `min_delta`: Evita despliegues innecesarios por mesetas exigiendo una mejora porcentual configurada (evita inestabilidad).
    * `per_class_f1`: Orientada a escenarios con alto desequilibrio de costes (como fraude bancario), activando el modelo solo si mejora una clase específica prioritaria.