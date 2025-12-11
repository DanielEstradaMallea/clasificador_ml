import os
import time
import json
import logging
import sys
import torch
import torch.quantization # Importante para la optimización
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURACIÓN DE LOGS (Observabilidad) ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

app = Flask(__name__)

# --- CARGA DEL MODELO ---
MODEL_PATH = "./models" 
device = torch.device("cpu") # Railway usa CPU por defecto

logger.info(json.dumps({"event": "startup", "message": f"Cargando modelo desde {MODEL_PATH}..."}))

# --- VALIDACIÓN DE ARCHIVO LFS ---
archivo_modelo = os.path.join(MODEL_PATH, "model.safetensors")
if os.path.exists(archivo_modelo):
    peso = os.path.getsize(archivo_modelo)
    logger.info(json.dumps({"event": "file_check", "file": "model.safetensors", "size_bytes": peso}))
    
    if peso < 10000: # Si pesa menos de 10KB, es un puntero LFS roto
        msg = f"ERROR CRÍTICO: El archivo del modelo pesa solo {peso} bytes. Es un puntero LFS, no el binario real."
        logger.error(json.dumps({"event": "lfs_error", "message": msg}))
        # No hacemos raise inmediato para permitir ver el log, pero fallará abajo.
else:
    logger.error(json.dumps({"event": "file_check", "error": "Archivo no encontrado"}))
# ---------------------------------

try:
    # 1. Cargar Tokenizer y Modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # 2. BLOQUE DE OPTIMIZACIÓN (QUANTIZATION)
    # Esto reduce drásticamente el peso de las operaciones matemáticas (de 32-bit a 8-bit)
    # Ideal para CPUs sin tarjeta de video.
    print("⚡ Optimizando modelo para CPU (Quantization)...")
    model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Aplicar solo a las capas lineales (las más pesadas)
        dtype=torch.qint8
    )
    
    model.to(device)
    model.eval() # Modo evaluación (desactiva randomness)
    
    logger.info(json.dumps({"event": "startup", "status": "success", "message": "Modelo cargado y CUANTIZADO (Optimizado)"}))

except Exception as e:
    logger.error(json.dumps({"event": "startup_error", "error": str(e)}))
    raise e

# --- ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    """Monitor de disponibilidad para Railway"""
    return jsonify({"status": "ok", "model_loaded": True}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # 1. Validación de entrada (Fail Fast)
    data = request.get_json(force=True, silent=True)
    if not data or 'texto' not in data:
        return jsonify({'error': 'Falta campo "texto" en el JSON'}), 400
    
    texto_entrada = data['texto']
    
    try:
        # 2. Preprocesamiento Optimizado
        # max_length=128 es suficiente para denuncias cortas y ahorra mucha CPU
        inputs = tokenizer(
            texto_entrada, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128 
        ).to(device)
        
        # 3. Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 4. Post-procesamiento
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        # Obtener la clase con mayor probabilidad
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # Mapear índice a etiqueta (usando tu config.json ya corregido)
        predicted_label = model.config.id2label[pred_idx]
        
        # Cálculo de latencia
        latency_ms = (time.time() - start_time) * 1000
        
        # 5. Logging Estructurado
        logger.info(json.dumps({
            "event": "inference",
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }))

        # 6. Respuesta al Cliente
        return jsonify({
            'clase': predicted_label,
            'probabilidad': round(confidence, 4)
        })

    except Exception as e:
        logger.error(json.dumps({"event": "inference_error", "error": str(e)}))
        return jsonify({'error': 'Error interno procesando la solicitud'}), 500

if __name__ == '__main__':
    # Ejecución local
    app.run(host='0.0.0.0', port=5000)