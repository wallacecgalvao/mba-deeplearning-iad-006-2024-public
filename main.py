from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Criação da aplicação FastAPI
app = FastAPI()

# Carregar o dataset Iris como exemplo
iris = load_iris()
X = iris.data
y = iris.target

# Treinar o modelo de árvore de decisão
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Modelo de dados para previsão
class PredictionInput(BaseModel):
    data: list

# Rota para verificação do status do servidor
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Rota para realizar previsões com o modelo
@app.post("/predict/")
def predict(input: PredictionInput):
    # Transformar a entrada em um numpy array
    input_data = np.array(input.data).reshape(1, -1)
    
    # Realizar a previsão
    prediction = clf.predict(input_data)
    
    # Retornar a classe prevista
    return {"prediction": int(prediction[0])}
