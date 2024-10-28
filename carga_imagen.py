import streamlit as st
import tempfile

import os
from dotenv import load_dotenv
from openslide_utils import load_slide_image, load_slide_region, get_best_level
import subprocess
if os.name == 'nt':
    load_dotenv()  # Cargar variables de entorno desde .env
    OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")
    from ctypes import cdll

    dll_path = os.path.join(OPENSLIDE_PATH, 'libopenslide-1.dll')
    cdll.LoadLibrary(dll_path)

if not os.path.exists('/usr/lib/x86_64-linux-gnu/libopenslide.so.0') and not os.name =='nt':
    subprocess.run([
        'sudo', 'apt-get', 'update', '-y', '&&',
        'sudo', 'apt-get', 'install', '-y', 'libopenslide-dev', 'openslide-tools', 'libjpeg-dev', 'libtiff-dev'
    ], check=True)

import openslide
# Subtítulo para la selección de imágenes
st.header("Selección de Imagen")

# Verificar si ya hay una imagen cargada en session_state
if 'uploaded_slide' in st.session_state:
    slide = st.session_state.uploaded_slide
    np_image, _ = load_slide_region(slide, get_best_level(slide))
    st.image(np_image, caption="Imagen cargada anteriormente", use_column_width=True)
    # Opción para cargar una nueva imagen
    if st.button("Cargar una nueva imagen"):
        del st.session_state['uploaded_slide']  # Eliminar la imagen de session_state
else:
    uploaded_file = st.file_uploader("Elegí un archivo", type=["tif", "tiff", "cvs"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            slide_path = temp_file.name

        # Intentar cargar la imagen de OpenSlide
        try:
            slide = open_slide(slide_path)
            img = load_slide_image(slide_path)
            np_image, _ = load_slide_region(slide, get_best_level(slide))

            # Guardamos la imagen cargada en session_state
            st.session_state['uploaded_slide'] = slide
            st.image(np_image, caption="Imagen cargada", use_column_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error al cargar la imagen: {e}")