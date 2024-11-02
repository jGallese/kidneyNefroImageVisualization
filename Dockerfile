FROM python:3.12

WORKDIR /app

COPY packages.txt /tmp/packages.txt


# Instalar paquetes desde packages.txt y otros paquetes adicionales
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libopenslide-dev \
    openslide-tools \
    libjpeg-dev \
    libtiff-dev \
    python3-opencv \
    git && \
    rm -rf /var/lib/apt/lists/* /tmp/packages.txt

# Clonar el repositorio
RUN git clone https://github.com/jGallese/kidneyNefroImageVisualization.git .

# Instalar dependencias de Python
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
