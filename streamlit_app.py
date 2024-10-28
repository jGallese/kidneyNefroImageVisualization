import streamlit as st


cargar_imagen_page = st.Page("carga_imagen.py", title="Carga Imagen", icon=":material/upload:")

analisis_componentes_conectados_page = st.Page("analisis_componentes_conectados.py",
                                               title="Analisis de componentes conectados",
                                               icon=":material/add_circle:")

analisis_normalizacion_page = st.Page("analisis_normalizacion.py", title="Analisis normalizacion", icon=":material/zoom_out_map:")
pg = st.navigation([cargar_imagen_page, analisis_componentes_conectados_page, analisis_normalizacion_page])
st.set_page_config(page_title="Análisis de Microscopía", page_icon=":material/biotech:")
pg.run()