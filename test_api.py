import requests
import time
import sys

# URL local para el entorno de pruebas de GitHub Actions
API_URL = "http://127.0.0.1:5000/predict"
HEALTH_URL = "http://127.0.0.1:5000/health"

# Casos de prueba críticos
TEST_CASES = [
    {"texto": "Me robaron el auto a punta de pistola en la entrada de mi casa."},
    {"texto": "me quitaron la mochila luego de botarme al suelo y pegarme con palos."},
    {"texto": "me golpearon y empujaron para robarme la mochila."},
    {"texto": "mi vecino vende sustancias ilicitas en su casa"}
    
]

def wait_for_server():
    """Espera a que el servidor arranque y descargue el modelo"""
    print("⏳ Esperando a que el servidor inicie...")
    for _ in range(30): # Intenta por 60 segundos (30x2)
        try:
            r = requests.get(HEALTH_URL)
            if r.status_code == 200:
                print("✅ Servidor listo!")
                return True
        except:
            pass
        time.sleep(2)
        sys.stdout.write(".")
        sys.stdout.flush()
    return False

def run_tests():
    if not wait_for_server():
        print("\n❌ Error: El servidor no inició a tiempo.")
        sys.exit(1) # Código de error para detener GitHub Actions

    fallos = 0
    print(f"\n🚀 Iniciando batería de pruebas contra: {API_URL}")

    for i, case in enumerate(TEST_CASES):
        print(f"\nCaso {i+1}: '{case['texto'][:40]}...'")
        try:
            start = time.time()
            response = requests.post(API_URL, json={"texto": case['texto']})
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                pred = data.get('clase', 'Desconocido')
                conf = data.get('probabilidad', 0)
                
                print(f"   Predicción: {pred} | Confianza: {conf:.1%} | Latencia: {latency:.0f}ms")
                
                # Validaciones de Calidad (Quality Gate)
                if conf < 0.50:
                    print("   ⚠️  Advertencia: Confianza muy baja.")
                    # fallos += 1  <-- Descomentar si quieres ser estricto
                
                if latency > 3000: # 3 segundos max en entorno de prueba
                     print("   ⚠️  Advertencia: Latencia alta.")
            else:
                print(f"   ❌ Error HTTP {response.status_code}")
                fallos += 1
                
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")
            fallos += 1

    if fallos > 0:
        print(f"\n❌ SE DETECTARON {fallos} FALLOS. EL DESPLIEGUE DEBE CANCELARSE.")
        sys.exit(1) # Esto pone el semáforo en ROJO en GitHub
    else:
        print("\n✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE.")
        sys.exit(0) # Esto pone el semáforo en VERDE

if __name__ == "__main__":
    run_tests()