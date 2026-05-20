import pandas as pd
from sklearn.datasets import load_iris
from datasets import Dataset

# 1. Cargar los datos locales
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

# 2. Convertir al formato exacto que pide Hugging Face
hf_dataset = Dataset.from_pandas(df)

# 3. Subir el dataset a la nube (¡Cambia el usuario aquí!)
hf_dataset.push_to_hub("CarlosSeoane/iris")

print("¡Dataset subido con éxito a Hugging Face!")