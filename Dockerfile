FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Cambiar el repositorio a un espejo alternativo
RUN sed -i 's/deb.debian.org/mirror.us.leaseweb.net/g' /etc/apt/sources.list

# Actualizar e instalar dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libopenslide-dev \
    openslide-tools \
    libjpeg-dev \
    libtiff-dev \
    python3-opencv \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copiar el resto de la aplicación
COPY . .

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar tu aplicación
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
