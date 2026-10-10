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
    df = load_dataset_df()
    
    test_results = metrics.get('model_evaluation', {}).get('test_results', [])
    test_labels = [item['periode'] for item in test_results]
    actual_vals = [item['aktual'] for item in test_results]
    pred_vals = [item['prediksi'] for item in test_results]
    
    # Data linimasa penuh (149 bulan)
    full_labels = df['periode'].tolist() if not df.empty else []
    full_actual = df['penjualan_unit'].tolist() if not df.empty else []
    
    # Isi prediksi hanya pada 6 bulan terakhir (posisi testing data)
    num_train = max(0, len(full_labels) - len(test_labels))
    full_pred = ([None] * num_train) + pred_vals
    
    return render_template('evaluasi.html', 
                           metrics=metrics,
                           test_labels=test_labels,
                           actual_vals=actual_vals,
                           pred_vals=pred_vals,
                           full_labels=full_labels,
                           full_actual=full_actual,
                           full_pred=full_pred)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi_page():
    metrics = load_metrics()
    artifacts = load_model_artifacts()
    df = load_dataset_df()
    
    # 10 data riil terakhir GAIKINDO sebagai nilai default form
    if not df.empty:
        default_inputs = df['penjualan_unit'].iloc[-10:].tolist()
        default_labels = df['periode'].iloc[-10:].tolist()
    else:
        default_inputs = [27239, 26367, 30844, 33449, 33740, 26667, 30632, 28970, 27336, 21518]
        default_labels = [f"Bulan {i+1}" for i in range(10)]
        
    current_inputs = list(default_inputs)
    result = None
    
    if request.method == 'POST':
        try:
            # Ambil 10 angka yang diinput manual oleh user/dosen
            manual_vals = []
            for i in range(1, 11):
                val = float(request.form.get(f'm{i}', default_inputs[i-1]))
                manual_vals.append(val)
                
            current_inputs = manual_vals
            
            if artifacts:
                mlp = artifacts['model']
                scaler_X = artifacts['scaler_X']
                scaler_y = artifacts['scaler_y']
                
                # Transformasi dan prediksi menggunakan MLP
                input_scaled = scaler_X.transform([manual_vals])
                pred_scaled = mlp.predict(input_scaled)
                pred_val = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                pred_int = int(round(pred_val))
                
                # Hitung tren kenaikan/penurunan dibanding bulan terakhir
                last_month_val = manual_vals[-1]
                diff = pred_int - last_month_val
                pct_change = round((diff / last_month_val) * 100, 2)
                
                # Siapkan data untuk grafik visual interaktif (10 input + 1 prediksi)
                chart_history = [int(v) for v in manual_vals]
                
                result = {
                    'prediction_val': pred_int,
                    'last_val': int(last_month_val),
                    'diff': int(diff),
                    'pct_change': pct_change,
                    'is_increase': diff >= 0,
                    'chart_history': chart_history,
                    'chart_labels': [f"Bulan {i}" for i in range(1, 11)] + ["Prediksi Bulan ke-11"]
                }
        except Exception as e:
            result = {'error': str(e)}
            
    return render_template('prediksi.html', 
                           metrics=metrics, 
                           current_inputs=current_inputs,
                           default_inputs=default_inputs,
                           result=result)

@app.route('/komparasi')
def komparasi_page():
    metrics = load_metrics()
    return render_template('komparasi.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
