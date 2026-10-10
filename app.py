import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Direktori base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'iris.csv')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')
DT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'decision_tree_model.pkl')
MLP_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mlp_model.pkl')

# Helper: Memuat metadata evaluasi
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {}

# Helper: Memuat model
dt_model = None
mlp_data = None

if os.path.exists(DT_MODEL_PATH):
    dt_model = joblib.load(DT_MODEL_PATH)

if os.path.exists(MLP_MODEL_PATH):
    mlp_data = joblib.load(MLP_MODEL_PATH)

# Helper: Memuat dataset DataFrame
def load_dataset_df():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()

# ----------------- ROUTING -----------------

@app.route('/')
def index():
    metrics = load_metrics()
    return render_template('index.html', metrics=metrics)

@app.route('/dataset')
def dataset_page():
    metrics = load_metrics()
    df = load_dataset_df()
    # Ambil seluruh baris data sebagai list of dict untuk preview di tabel
    records = df.to_dict(orient='records') if not df.empty else []
    return render_template('dataset.html', metrics=metrics, data=records, total_rows=len(records))

@app.route('/download-dataset')
def download_dataset():
    if os.path.exists(DATASET_PATH):
        return send_file(DATASET_PATH, as_attachment=True, download_name='iris.csv')
    return "File tidak ditemukan", 404

@app.route('/evaluasi')
def evaluasi_page():
    metrics = load_metrics()
    return render_template('evaluasi.html', metrics=metrics)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi_page():
    result = None
    input_data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2,
        'algorithm': 'dt'
    }
    
    if request.method == 'POST':
        try:
            sl = float(request.form.get('sepal_length', 5.1))
            sw = float(request.form.get('sepal_width', 3.5))
            pl = float(request.form.get('petal_length', 1.4))
            pw = float(request.form.get('petal_width', 0.2))
            algorithm = request.form.get('algorithm', 'dt')
            
            input_data = {
                'sepal_length': sl,
                'sepal_width': sw,
                'petal_length': pl,
                'petal_width': pw,
                'algorithm': algorithm
            }
            
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            features_df = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_cols)
            classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            
            if algorithm == 'mlp' and mlp_data is not None:
                model = mlp_data['model']
                scaler = mlp_data['scaler']
                scaled_features = scaler.transform(features_df)
                pred_label = model.predict(scaled_features)[0]
                proba = model.predict_proba(scaled_features)[0]
                algo_title = "Multilayer Perceptron (MLP)"
            else:
                pred_label = dt_model.predict(features_df)[0]
                proba = dt_model.predict_proba(features_df)[0]
                algo_title = "Decision Tree (CART Gini)"
            
            # Siapkan info spesies
            species_info = {
                'Iris-setosa': {
                    'name': 'Iris Setosa',
                    'image': 'setosa.jpg',
                    'desc': 'Memiliki ukuran mahkota (petal) yang pendek dan lebar, serta kelopak (sepal) yang proporsional. Sangat mudah dipisahkan secara linier dari spesies lain.',
                    'badge_class': 'badge-setosa'
                },
                'Iris-versicolor': {
                    'name': 'Iris Versicolor',
                    'image': 'versicolor.jpg',
                    'desc': 'Merupakan spesies berukuran sedang dengan kelopak dan mahkota yang seimbang. Memiliki kemiripan ukuran dengan Iris Virginica di beberapa rentang nilai.',
                    'badge_class': 'badge-versicolor'
                },
                'Iris-virginica': {
                    'name': 'Iris Virginica',
                    'image': 'virginica.jpg',
                    'desc': 'Memiliki ukuran mahkota (petal) yang paling panjang dan lebar di antara ketiga spesies. Ciri utamanya tampak pada petal length > 4.8 cm.',
                    'badge_class': 'badge-virginica'
                }
            }
            
            probabilities = [
                {'class_name': classes[i], 'label': species_info[classes[i]]['name'], 'score': round(float(proba[i]) * 100, 1)}
                for i in range(len(classes))
            ]
            
            result = {
                'prediction': pred_label,
                'info': species_info.get(pred_label, species_info['Iris-setosa']),
                'algorithm_used': algo_title,
                'probabilities': probabilities,
                'confidence': round(float(np.max(proba)) * 100, 1)
            }
        except Exception as e:
            result = {'error': str(e)}
            
    return render_template('prediksi.html', input_data=input_data, result=result)

@app.route('/komparasi')
def komparasi_page():
    metrics = load_metrics()
    return render_template('komparasi.html', metrics=metrics)

# API endpoint untuk prediksi via AJAX (opsional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    try:
        sl = float(data.get('sepal_length', 5.1))
        sw = float(data.get('sepal_width', 3.5))
        pl = float(data.get('petal_length', 1.4))
        pw = float(data.get('petal_width', 0.2))
        algorithm = data.get('algorithm', 'dt')
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        features_df = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_cols)
        classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        if algorithm == 'mlp' and mlp_data is not None:
            model = mlp_data['model']
            scaler = mlp_data['scaler']
            scaled_features = scaler.transform(features_df)
            pred_label = model.predict(scaled_features)[0]
            proba = model.predict_proba(scaled_features)[0].tolist()
        else:
            pred_label = dt_model.predict(features_df)[0]
            proba = dt_model.predict_proba(features_df)[0].tolist()
            
        return jsonify({
            'status': 'success',
            'prediction': pred_label,
            'probabilities': {classes[i]: round(proba[i] * 100, 2) for i in range(len(classes))}
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
