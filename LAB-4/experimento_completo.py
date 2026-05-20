import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Guardamos el test para luego
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'class_weight': ['balanced', None]
}

grid = ParameterGrid(param_grid)
mlflow.set_experiment("Diabetes_Experimento_Completo")

best_recall = 0
best_run_id = None

for params in grid:
    with mlflow.start_run():
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        mlflow.sklearn.log_model(model, "rf_model")
        
        if rec > best_recall:
            best_recall = rec
            best_run_id = mlflow.active_run().info.run_id

print(f"EL MEJOR MODELO TIENE EL ID: {best_run_id}")