# 1. Imagen base oficial de Python (Estable)
FROM python:3.10-slim

# 2. Instalar dependencias del sistema y Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs gcc && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# 3. Configurar directorio de trabajo
WORKDIR /app

# 4. Copiar archivos de requerimientos e instalar librerías
COPY requirements.txt .
# Actualizamos pip e instalamos las dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código (incluyendo la carpeta models)
COPY . .

# NOTA CRÍTICA:
# Docker copiará los archivos tal como los descargó Railway desde GitHub.
# Si Railway descargó punteros, aquí tendremos punteros.
# Intentaremos forzar un 'lfs pull' por si acaso el contexto .git existe.
RUN git lfs install || true
RUN git lfs pull || true

# 6. Exponer el puerto y comando de arranque
ENV PORT=8080
EXPOSE 8080

# Usamos un timeout alto para dar tiempo a la carga del modelo
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "--log-file", "-"]