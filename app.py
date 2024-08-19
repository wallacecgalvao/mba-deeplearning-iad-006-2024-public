from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# Carregar o dataset Iris como exemplo
iris = load_iris()
X = iris.data
y = iris.target

# Treinar o modelo de árvore de decisão com os melhores parâmetros
clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=2)
clf.fit(X, y)

@app.route('/')
def home():
    return "API is running using Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    input_data = np.array(data).reshape(1, -1)
    prediction = clf.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
