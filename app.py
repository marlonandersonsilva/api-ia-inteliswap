from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Verificar se os modelos existem e carregá-los
model_path = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_path, exist_ok=True)

try:
    modelo_rf = joblib.load(os.path.join(model_path, 'modelo_random_forest.pkl'))
    print("Modelo Random Forest carregado com sucesso!")
except FileNotFoundError:
    print("Aviso: modelo_random_forest.pkl não encontrado. Endpoint /predict_rf não funcionará corretamente.")
    modelo_rf = None

try:
    modelo_xgb = joblib.load(os.path.join(model_path, 'modelo_xgboost.pkl'))
    print("Modelo XGBoost carregado com sucesso!")
except FileNotFoundError:
    print("Aviso: modelo_xgboost.pkl não encontrado. Endpoint /predict_xgb não funcionará corretamente.")
    modelo_xgb = None

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    if modelo_rf is None:
        return jsonify({'erro': 'Modelo Random Forest não está disponível'}), 503
    
    try:
        dados = request.get_json()
        if not dados or 'features' not in dados:
            return jsonify({'erro': 'Dados de entrada inválidos. Forneça um JSON com o campo "features"'}), 400
        
        features = np.array(dados['features']).reshape(1, -1)
        previsao = modelo_rf.predict(features)
        return jsonify({'previsao': previsao.tolist()})
    except Exception as e:
        return jsonify({'erro': f'Erro ao processar a previsão: {str(e)}'}), 500

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    if modelo_xgb is None:
        return jsonify({'erro': 'Modelo XGBoost não está disponível'}), 503
    
    try:
        dados = request.get_json()
        if not dados or 'features' not in dados:
            return jsonify({'erro': 'Dados de entrada inválidos. Forneça um JSON com o campo "features"'}), 400
        
        features = np.array(dados['features']).reshape(1, -1)
        previsao = modelo_xgb.predict(features)
        return jsonify({'previsao': previsao.tolist()})
    except Exception as e:
        return jsonify({'erro': f'Erro ao processar a previsão: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'online',
        'modelos': {
            'random_forest': modelo_rf is not None,
            'xgboost': modelo_xgb is not None
        }
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)