import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'penjualan_toyota_gaikindo.csv')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mlp_time_series.pkl')

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {}

def load_model_artifacts():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def load_dataset_df():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()

# ----------------- ROUTING -----------------

@app.route('/')
def index():
    metrics = load_metrics()
    df = load_dataset_df()
    
    # Siapkan data untuk grafik historis di beranda
    chart_labels = df['periode'].tolist() if not df.empty else []
    chart_sales = df['penjualan_unit'].tolist() if not df.empty else []
    
    return render_template('index.html', 
                           metrics=metrics, 
                           chart_labels=chart_labels, 
                           chart_sales=chart_sales)

@app.route('/dataset')
def dataset_page():
    metrics = load_metrics()
    df = load_dataset_df()
    records = df.to_dict(orient='records') if not df.empty else []
    return render_template('dataset.html', metrics=metrics, data=records, total_rows=len(records))

@app.route('/download-dataset')
def download_dataset():
    if os.path.exists(DATASET_PATH):
        return send_file(DATASET_PATH, as_attachment=True, download_name='penjualan_toyota_gaikindo.csv')
    return "File tidak ditemukan", 404

@app.route('/evaluasi')
def evaluasi_page():
    metrics = load_metrics()
    
    # Ambil data evaluasi test set untuk grafik
    test_results = metrics.get('model_evaluation', {}).get('test_results', [])
    test_labels = [item['periode'] for item in test_results]
    actual_vals = [item['aktual'] for item in test_results]
    pred_vals = [item['prediksi'] for item in test_results]
    
    return render_template('evaluasi.html', 
                           metrics=metrics,
                           test_labels=test_labels,
                           actual_vals=actual_vals,
                           pred_vals=pred_vals)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi_page():
    metrics = load_metrics()
    artifacts = load_model_artifacts()
    df = load_dataset_df()
    
    future_labels_all = ['2023-06 (Juni 2023)', '2023-07 (Juli 2023)', '2023-08 (Agustus 2023)', 
                         '2023-09 (September 2023)', '2023-10 (Oktober 2023)', '2023-11 (November 2023)']
    
    selected_steps = 1
    forecast_results = None
    
    if request.method == 'POST':
        try:
            selected_steps = int(request.form.get('horizon', 1))
        except ValueError:
            selected_steps = 1
            
        if artifacts:
            mlp = artifacts['model']
            scaler_X = artifacts['scaler_X']
            scaler_y = artifacts['scaler_y']
            current_window = list(artifacts['last_window'])
            
            forecast_items = []
            for i in range(selected_steps):
                input_scaled = scaler_X.transform([current_window])
                pred_s = mlp.predict(input_scaled)
                pred_val = float(scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0][0])
                pred_int = int(round(pred_val))
                
                forecast_items.append({
                    'step': i + 1,
                    'periode': future_labels_all[i],
                    'prediksi_unit': pred_int
                })
                
                # geser window
                current_window.pop(0)
                current_window.append(pred_val)
                
            forecast_results = forecast_items
            
    # Default 1 bulan jika belum submit
    if forecast_results is None and metrics.get('forecast_future'):
        forecast_results = [{
            'step': 1,
            'periode': future_labels_all[0],
            'prediksi_unit': metrics['forecast_future']['predictions'][0]
        }]
        
    return render_template('prediksi.html', 
                           metrics=metrics, 
                           selected_steps=selected_steps,
                           forecast_results=forecast_results)

@app.route('/komparasi')
def komparasi_page():
    metrics = load_metrics()
    return render_template('komparasi.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
