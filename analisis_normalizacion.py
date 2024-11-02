import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib import pyplot as plt
import os
import cv2
from openslide_utils import load_slide_image, load_slide_region, get_best_level
from normalization_utils import norm_Masson_modified as norm_Masson


# Caches y funciones de utilidad
@st.cache_resource
def initialize_slide_view(_slide):
    """
    Inicializa la vista del slide y mantiene los recursos en caché.
    """
    best_level = _slide.get_best_level_for_downsample(16)
    level_dimensions = _slide.level_dimensions[best_level]
    image = _slide.read_region((0, 0), best_level, level_dimensions)
    return best_level, level_dimensions, np.array(image.convert("RGB"))


@st.cache_data
def get_initial_region(_slide):
    """
    Obtiene la región inicial de análisis.
    """
    region = _slide.read_region((10000, 4000), 0, (5000, 5000))
    return np.array(region.convert('RGB'))


@st.cache_data
def get_region_image(_slide, coords):
    """
    Obtiene una región específica de la imagen.
    """
    top_left_x_0, top_left_y_0, region_width, region_height = coords
    region = _slide.read_region((top_left_x_0, top_left_y_0), 0, (region_width, region_height))
    return np.array(region.convert('RGB'))


@st.cache_data
def perform_masson_normalization(image_np, tr_ref_values, max_c_ref, io=240, alpha=1, beta=0.20):
    """
    Realiza la normalización de Masson con los parámetros dados.
    """
    TRRef = np.array([
        [tr_ref_values["00"], tr_ref_values["01"]],
        [tr_ref_values["10"], tr_ref_values["11"]],
        [tr_ref_values["20"], tr_ref_values["21"]]
    ])
    return norm_Masson(image_np, TRRef, max_c_ref, Io=io, alpha=alpha, beta=beta)


@st.cache_data
def process_image_masks(image, min_val, max_val):
    """
    Procesa las máscaras de la imagen.
    """
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.inRange(normalized_img, min_val, max_val)
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return normalized_img, mask, result


def display_images(original, norm, collagen, other, title=""):
    """
    Muestra las imágenes en un layout de 2 columnas.
    """
    if title:
        st.subheader(title)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_column_width=True)
        st.image(collagen, caption="Collagen Image", use_column_width=True)
    with col2:
        st.image(norm, caption="Normalized Image", use_column_width=True)
        st.image(other, caption="Other Things Image", use_column_width=True)


def create_sidebar_controls():
    """
    Crea y maneja los controles del sidebar.
    """
    st.sidebar.header("Ajustes de TRRef")

    tr_ref_values = {
        "00": st.sidebar.slider("Coeficiente Colágeno (1→1)", 0.0, 1.0, 0.7, 0.01),
        "01": st.sidebar.slider("Coeficiente Colorante (1→2)", 0.0, 1.0, 0.1, 0.01),
        "10": st.sidebar.slider("Coeficiente Otro (2→1)", 0.0, 1.0, 0.15, 0.01),
        "11": st.sidebar.slider("Coeficiente Otro (2→2)", 0.0, 1.0, 0.9, 0.01),
        "20": st.sidebar.slider("Coeficiente Tercer (3→1)", 0.0, 1.0, 0.2, 0.01),
        "21": st.sidebar.slider("Coeficiente Tercer (3→2)", 0.0, 1.0, 0.8, 0.01)
    }

    st.sidebar.header("Ajustes de Concentración")
    max_c_ref = np.array([
        st.sidebar.slider("Máx. colágeno", 0.0, 5.0, 2.0, 0.1),
        st.sidebar.slider("Máx. otro colorante", 0.0, 5.0, 1.0, 0.1)
    ])

    st.sidebar.header("Ajustes de Umbral")
    min_val = st.sidebar.slider("Valor mínimo", 0, 255, 50)
    max_val = st.sidebar.slider("Valor máximo", 0, 255, 150)

    return tr_ref_values, max_c_ref, min_val, max_val


def process_canvas_selection(canvas_result, slide, best_level, level_dimensions):
    """
    Procesa la selección del canvas y retorna la región seleccionada.
    """
    if not canvas_result.json_data:
        return None

    for shape in canvas_result.json_data["objects"]:
        if shape["type"] != "rect":
            continue

        # Obtener coordenadas
        left, top = int(shape["left"]), int(shape["top"])
        width, height = int(shape["width"]), int(shape["height"])

        # Calcular escalas
        scale_x = level_dimensions[0] / 704  # canvas_width
        scale_y = level_dimensions[1] / 200  # canvas_height

        # Calcular coordenadas en diferentes niveles
        top_left_x = int(left * scale_x)
        top_left_y = int(top * scale_y)
        bottom_right_x = int((left + width) * scale_x)
        bottom_right_y = int((top + height) * scale_y)

        scale_factor = slide.level_downsamples[best_level]
        top_left_x_0 = int(top_left_x * scale_factor)
        top_left_y_0 = int(top_left_y * scale_factor)
        bottom_right_x_0 = int(bottom_right_x * scale_factor)
        bottom_right_y_0 = int(bottom_right_y * scale_factor)

        coords = (
            top_left_x_0,
            top_left_y_0,
            bottom_right_x_0 - top_left_x_0,
            bottom_right_y_0 - top_left_y_0
        )

        return {
            "coords": coords,
            "levels": {
                "best": (top_left_x, top_left_y, bottom_right_x, bottom_right_y),
                "zero": (top_left_x_0, top_left_y_0, bottom_right_x_0, bottom_right_y_0)
            }
        }

    return None


def main():
    st.header("Análisis de Normalización")

    if 'uploaded_slide' not in st.session_state:
        st.markdown("---")
        st.header("Por favor, selecciona una imagen para continuar")
        st.markdown("---")
        return

    slide = st.session_state.uploaded_slide

    # Inicializar vista y controles
    best_level, level_dimensions, background_image = initialize_slide_view(slide)
    tr_ref_values, max_c_ref, min_val, max_val = create_sidebar_controls()


    # Configuración del canvas
    st.title("Selecciona un punto en la imagen")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=Image.fromarray(background_image),
        update_streamlit=True,
        width=704,
        height=200,
        drawing_mode="rect",
        point_display_radius=20,
        key="canvas",
    )

    # Procesar selección del canvas
    selection = process_canvas_selection(canvas_result, slide, best_level, level_dimensions)
    if selection:
        st.write(f"Coordenadas en nivel {best_level}:")
        st.write(f"Esquina superior izquierda: {selection['levels']['best'][:2]}")
        st.write(f"Esquina inferior derecha: {selection['levels']['best'][2:]}")
        st.write("Coordenadas en nivel 0:")
        st.write(f"Esquina superior izquierda: {selection['levels']['zero'][:2]}")
        st.write(f"Esquina inferior derecha: {selection['levels']['zero'][2:]}")

        # Obtener y procesar región seleccionada
        region_image_np = get_region_image(slide, selection["coords"])
        norm_img, collagen_img, other_img = perform_masson_normalization(
            region_image_np, tr_ref_values, max_c_ref
        )
        display_images(
            region_image_np, norm_img, collagen_img, other_img,
            "Análisis de Región Seleccionada"
        )

