import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Cargar los datos
train = pd.read_csv('train (3).csv')
test = pd.read_csv('test (2).csv')

# 2. Separar características (X) y la variable a predecir (y)
X = train.drop(['id', 'failure'], axis=1)
y = train['failure']
X_test = test.drop(['id'], axis=1)

# Identificar columnas numéricas y categóricas
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

# Unir train y test temporalmente para aplicar las mismas transformaciones
all_data = pd.concat([X, X_test], keys=['train', 'test'])

# 3. Limpieza y preparación de datos (Imputación)
# Rellenar valores nulos numéricos con la mediana
num_imputer = SimpleImputer(strategy='median')
all_data[num_cols] = num_imputer.fit_transform(all_data[num_cols])

# Convertir variables categóricas (texto a números)
for col in cat_cols:
    all_data[col] = all_data[col].fillna('missing')
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col])

# Separar de nuevo en train y test
X_clean = all_data.xs('train')
X_test_clean = all_data.xs('test')

# 4. Dividir el conjunto de entrenamiento para validar (80% entrena, 20% valida)
X_train, X_val, y_train, y_val = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# 5. Entrenar el modelo (Bosque Aleatorio)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluar el modelo (ROC-AUC)
y_val_pred = model.predict_proba(X_val)[:, 1] # Tomar solo la probabilidad de la clase 1 (fallo)
auc = roc_auc_score(y_val, y_val_pred)
print(f"Rendimiento del modelo en validación (ROC-AUC): {auc:.4f}")

# 7. Generar predicciones para el archivo de prueba (Test)
test_preds = model.predict_proba(X_test_clean)[:, 1]

# 8. Guardar los resultados en el formato solicitado
submission = pd.DataFrame({
    'id': test['id'], 
    'failure': test_preds
})
submission.to_csv('submission.csv', index=False)
print("¡Archivo 'submission.csv' creado con éxito!")