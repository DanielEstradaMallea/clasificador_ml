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

# --- CONFIGURACIÓN DEL MODELO ---
# Tu repositorio en Hugging Face
MODEL_ID = "RuloDan/clasificador-seguridad-xyz"

# Token de seguridad (necesario si tu repo es PRIVADO)
# Si es público, esta variable puede estar vacía y funcionará igual.
hf_token = os.environ.get("HF_TOKEN")

device = torch.device("cpu") # Railway usa CPU por defecto

logger.info(json.dumps({"event": "startup", "message": f"Iniciando descarga de {MODEL_ID}..."}))

try:
    # 1. Cargar Tokenizer y Modelo desde la Nube (Hugging Face Hub)
    # Transformers gestiona la descarga y el caché automáticamente.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=hf_token)
    
    # 2. BLOQUE DE OPTIMIZACIÓN (QUANTIZATION)
    # Reduce el peso matemático de 32-bit a 8-bit para velocidad en CPU
    print("⚡ Optimizando modelo para CPU (Quantization)...")
    model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Optimizar capas lineales
        dtype=torch.qint8
    )
    
    model.to(device)
    model.eval() # Modo evaluación
    
    logger.info(json.dumps({"event": "startup", "status": "success", "message": "Modelo descargado de HF y optimizado"}))

except Exception as e:
    logger.error(json.dumps({"event": "startup_error", "error": str(e)}))
    # Es crítico que falle aquí si no puede descargar el modelo
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
        inputs = tokenizer(
            texto_entrada, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128 # Límite para ahorrar CPU
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
        
        # Mapear índice a etiqueta (usando la config descargada de HF)
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
    app.run(host='0.0.0.0', port=5000)