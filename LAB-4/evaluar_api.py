import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

data_json = {"dataframe_split": X_test.to_dict(orient="split")}

url = "http://127.0.0.1:5000/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data_json, headers=headers)

if response.status_code == 200:
    predicciones = response.json()["predictions"]
    
    acc = accuracy_score(y_test, predicciones)
    prec = precision_score(y_test, predicciones)
    rec = recall_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)
    
    print("MÉTRICAS REALES DE LA API:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
else:
    print("Error al conectar con la API")