from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Carregar o modelo treinado
model = xgb.XGBClassifier()
model.load_model("xgboost_mnist_8x8.json")

# Função para processar a imagem e realizar a previsão
def process_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Converte para escala de cinza
    image = image.resize((8, 8))  # Garante que a imagem seja de 8x8 pixels
    image_array = np.array(image).flatten().astype('float32')  # Flatten para um vetor 1D
    image_array = image_array.reshape(1, -1)  # Redimensiona para o formato esperado pelo modelo
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image'].read()
    image_data = process_image(image_file)
    prediction = model.predict(image_data)
    
    return jsonify({'prediction': int(prediction[0])})

@app.route('/')
def home():
    return "XGBoost MNIST 8x8 Model API"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
