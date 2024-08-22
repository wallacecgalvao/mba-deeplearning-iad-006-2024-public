from fastapi import FastAPI
import xgboost as xgb
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port
"""        
app = FastAPI()

# Carregar o modelo (assumindo que você já treinou e salvou o modelo)
model = xgb.XGBClassifier()
model.load_model("xgboost_mnist_8x8.json")

# Classe para os dados de entrada
class DataInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: DataInput):
    data = np.array(input_data.data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "XGBoost MNIST model is ready for predictions!"}