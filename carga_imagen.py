import streamlit as st
import tempfile
import os
from openslide_utils import load_slide_image, load_slide_region, get_best_level
import openslide
from openslide import open_slide


# Cache para cargar el slide
@st.cache_resource
def load_slide_from_path(slide_path):
    """
    Carga un slide desde un archivo y lo mantiene en caché.
    """
    return open_slide(slide_path)


# Cache para procesar la imagen
@st.cache_data
def process_slide_image(slide):
    """
    Procesa la imagen del slide y devuelve la versión numpy.
    """
    best_level = get_best_level(slide)
    return load_slide_region(slide, best_level)


# Cache para manejar el archivo temporal
@st.cache_resource
def create_temp_file(file_content):
    """
    Crea un archivo temporal y devuelve su ruta.
    Mantiene el archivo en caché para evitar recrearlo.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_slide.tif")

    with open(temp_path, 'wb') as f:
        f.write(file_content.getbuffer())

    return temp_path


def main():
    st.header("Selección de Imagen")

    if 'uploaded_slide' in st.session_state:
        slide = st.session_state.uploaded_slide
        np_image, image_size = process_slide_image(slide)
        st.image(np_image, caption="Imagen cargada anteriormente", use_column_width=True)

        # Opción para cargar una nueva imagen
        if st.button("Cargar una nueva imagen"):
            # Limpiar todos los caches relacionados
            create_temp_file.clear()
            process_slide_image.clear()
            del st.session_state['uploaded_slide']
            st.experimental_rerun()

    else:
        uploaded_file = st.file_uploader(
            "Elegí un archivo",
            type=["tif", "tiff", "cvs"],
            help="Selecciona un archivo en formato TIF, TIFF o CVS"
        )

        if uploaded_file is not None:
            try:
                # Crear archivo temporal cacheado
                temp_path = create_temp_file(uploaded_file)

                # Cargar slide cacheado
                slide = load_slide_from_path(temp_path)

                # Procesar imagen cacheada
                np_image, image_size = process_slide_image(slide)

                # Guardar en session_state
                st.session_state['uploaded_slide'] = slide

                # Mostrar imagen
                st.image(
                    np_image,
                    caption=f"Imagen cargada (tamaño: {image_size})",
                    use_column_width=True
                )

            except Exception as e:
                st.error(
                    "Error al cargar la imagen",
                    help=f"Detalles del error: {str(e)}"
                )
                # Limpiar caches en caso de error
                create_temp_file.clear()
                process_slide_image.clear()


if __name__ == "__main__":
    main()