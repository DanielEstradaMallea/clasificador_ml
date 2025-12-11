import requests
import time
import sys

# --- CONFIGURACIÓN ---
# Cambia esto cuando tengas tu URL de Railway (ej: https://mi-app.up.railway.app/predict)
API_URL = "https://clasificadorml-production.up.railway.app/predict" 

# Datos de prueba
TEST_CASES = [
    {"texto": "Me robaron el auto a punta de pistola en la entrada de mi casa."},
    {"texto": "me quitaron la mochila luego de botarme al suelo y pegarme con palos."},
    {"texto": "me golpearon y empujaron para robarme la mochila."},
    {"texto": "mi vecino vende sustancias ilicitas en su casa"}
    
]

def run_tests():
    print(f"--- Iniciando pruebas contra: {API_URL} ---")
    
    # 1. Prueba de Salud
    try:
        health = requests.get(API_URL.replace("/predict", "/health"))
        if health.status_code == 200:
            print("API Online (/health)")
        else:
            print(f"❌ Error en health check: {health.status_code}")
            return
    except Exception as e:
        print(f"❌ No se pudo conectar al servidor. ¿Está corriendo 'app.py'?\nError: {e}")
        return

    # 2. Pruebas de Predicción
    for i, case in enumerate(TEST_CASES):
        print(f"\nCaso {i+1}: '{case['texto'][:40]}...'")
        start = time.time()
        
        try:
            response = requests.post(API_URL, json=case)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Predicción: {data.get('clase')} | Confianza: {data.get('probabilidad'):.2%}")
                print(f"   Latencia: {latency:.0f}ms")
            else:
                print(f"   ❌ Error HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")

if __name__ == "__main__":
    run_tests()