from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib
from huggingface_hub import HfApi

# 1. Cargar los datos y dividirlos
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

# 2. Entrenar el modelo (Árbol de decisión)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Guardar el modelo en un archivo en tu ordenador
joblib.dump(model, "modelo_iris.joblib")

# 4. Crear el repositorio en Hugging Face y subir el archivo
api = HfApi()
repo_id = "CarlosSeoane/iris-dt"

print(f"Creando repositorio {repo_id} y subiendo modelo...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="modelo_iris.joblib",
    path_in_repo="modelo_iris.joblib",
    repo_id=repo_id,
)

print(f"¡Modelo subido con éxito a https://huggingface.co/{repo_id} !")