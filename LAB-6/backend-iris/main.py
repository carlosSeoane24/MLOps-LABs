from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

app = FastAPI()

# Descarga el modelo de tu Hugging Face (cambia por tu usuario si hace falta)
model_path = hf_hub_download(repo_id="CarlosSeoane/iris-dt", filename="modelo_iris.joblib")
model = joblib.load(model_path)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: IrisData):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)[0]
    classes = ["setosa", "versicolor", "virginica"]
    return {"class_name": classes[prediction]}