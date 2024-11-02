import streamlit as st
import os
from dotenv import load_dotenv
import subprocess
import openslide

# cargar_imagen_page = st.Page("carga_imagen.py", title="Carga Imagen", icon=":material/upload:")
#
# analisis_componentes_conectados_page = st.Page("analisis_componentes_conectados.py",
#                                                title="Analisis de componentes conectados",
#                                                icon=":material/add_circle:")
#
# analisis_normalizacion_page = st.Page("analisis_normalizacion.py", title="Analisis normalizacion", icon=":material/zoom_out_map:")
# pg = st.navigation([cargar_imagen_page, analisis_componentes_conectados_page, analisis_normalizacion_page])
# st.set_page_config(page_title="Análisis de Microscopía", page_icon=":material/biotech:")
# pg.run()


# Importar las páginas de la aplicación
from carga_imagen import main as cargar_imagen_main
from analisis_componentes_conectados import main as analisis_componentes_conectados_main
from analisis_normalizacion import main as analisis_normalizacion_main

# Configuración de la aplicación
st.set_page_config(page_title="Análisis de Microscopía", page_icon=":material/biotech:")

# Definir las páginas de navegación
cargar_imagen_page = st.Page("carga_imagen.py", title="Carga Imagen", icon=":material/upload:")
analisis_componentes_conectados_page = st.Page("analisis_componentes_conectados.py",
                                               title="Análisis de Componentes Conectados",
                                               icon=":material/add_circle:")
analisis_normalizacion_page = st.Page("analisis_normalizacion.py", title="Análisis Normalización", icon=":material/zoom_out_map:")

# Crear el menú de navegación
pg = st.navigation([cargar_imagen_page, analisis_componentes_conectados_page, analisis_normalizacion_page])

# Ejecutar la página seleccionada
if pg == cargar_imagen_page:
    cargar_imagen_main()
elif pg == analisis_componentes_conectados_page:
    analisis_componentes_conectados_main()
elif pg == analisis_normalizacion_page:
    analisis_normalizacion_main()