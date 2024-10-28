import streamlit as st
import os
from dotenv import load_dotenv
import subprocess
if os.name == 'nt':
    load_dotenv()  # Cargar variables de entorno desde .env
    OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")
    from ctypes import cdll

    dll_path = os.path.join(OPENSLIDE_PATH, 'libopenslide-1.dll')
    cdll.LoadLibrary(dll_path)

if not os.path.exists('/usr/lib/x86_64-linux-gnu/libopenslide.so.0'):
    subprocess.run([
        'sudo', 'apt-get', 'update', '-y', '&&',
        'sudo', 'apt-get', 'install', '-y', 'libopenslide-dev', 'openslide-tools', 'libjpeg-dev', 'libtiff-dev'
    ])

import openslide
cargar_imagen_page = st.Page("carga_imagen.py", title="Carga Imagen", icon=":material/upload:")

analisis_componentes_conectados_page = st.Page("analisis_componentes_conectados.py",
                                               title="Analisis de componentes conectados",
                                               icon=":material/add_circle:")

analisis_normalizacion_page = st.Page("analisis_normalizacion.py", title="Analisis normalizacion", icon=":material/zoom_out_map:")
pg = st.navigation([cargar_imagen_page, analisis_componentes_conectados_page, analisis_normalizacion_page])
st.set_page_config(page_title="Análisis de Microscopía", page_icon=":material/biotech:")
pg.run()