import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1) 
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Diabetes_Prueba")

with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(model, "modelo_prueba")
    print(f"¡Prueba superada! El ID de tu modelo es: {mlflow.active_run().info.run_id}")