import os
import time
import json
import logging
import sys
import torch
import torch.quantization
import torch.nn.functional as F
# 1. IMPORTAR render_template
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURACIÓN DE LOGS ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

app = Flask(__name__)
CORS(app) # Ya no es estrictamente necesario si usas el mismo dominio, pero es bueno dejarlo.

# --- CONFIGURACIÓN DEL MODELO ---
MODEL_ID = "RuloDan/clasificador-seguridad-xyz"
hf_token = os.environ.get("HF_TOKEN")
device = torch.device("cpu")

logger.info(json.dumps({"event": "startup", "message": f"Iniciando descarga de {MODEL_ID}..."}))

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=hf_token)
    
    print("⚡ Optimizando modelo para CPU (Quantization)...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.to(device)
    model.eval()
    logger.info(json.dumps({"event": "startup", "status": "success", "message": "Modelo listo"}))

except Exception as e:
    logger.error(json.dumps({"event": "startup_error", "error": str(e)}))
    raise e

# --- ENDPOINTS ---

# 2. NUEVA RUTA PARA SERVIR EL HTML
@app.route('/')
def home():
    """Ruta raíz que entrega el frontend"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": True}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Validación
    data = request.get_json(force=True, silent=True)
    if not data or 'texto' not in data:
        return jsonify({'error': 'Falta campo "texto" en el JSON'}), 400
    
    texto_entrada = data['texto']
    
    try:
        # Preprocesamiento
        inputs = tokenizer(
            texto_entrada, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-procesamiento
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        predicted_label = model.config.id2label[pred_idx]
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Logging
        logger.info(json.dumps({
            "event": "inference",
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }))

        return jsonify({
            'clase': predicted_label,
            'probabilidad': round(confidence, 4)
        })

    except Exception as e:
        logger.error(json.dumps({"event": "inference_error", "error": str(e)}))
        return jsonify({'error': 'Error interno procesando la solicitud'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)