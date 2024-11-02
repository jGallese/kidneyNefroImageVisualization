import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib import pyplot as plt
import os
import cv2
from openslide_utils import load_slide_image, load_slide_region, get_best_level
from normalization_utils import norm_Masson_modified as norm_Masson


# Cache para cargar regiones de la imagen
@st.cache_data
def get_region_image(_slide, top_left_x_0, top_left_y_0, region_width, region_height):
    return _slide.read_region((top_left_x_0, top_left_y_0), 0, (region_width, region_height)).convert('RGB')


# Cache para la normalización de Masson
@st.cache_data
def cached_norm_Masson(image_np, TRRef, maxCRef, Io, alpha, beta):
    return norm_Masson(image_np, TRRef, maxCRef, Io=Io, alpha=alpha, beta=beta)


# Cache para procesar imagen y obtener máscaras
@st.cache_data
def process_image_masks(image, min_val, max_val):
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.inRange(normalized_img, min_val, max_val)
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return normalized_img, mask, result


# Cache para obtener la mejor visualización del slide
@st.cache_resource
def get_slide_visualization(_slide):
    best_level = _slide.get_best_level_for_downsample(16)
    level_dimensions = _slide.level_dimensions[best_level]
    image = _slide.read_region((0, 0), best_level, level_dimensions)
    image_np = np.array(image.convert("RGB"))
    return image_np, best_level, level_dimensions


if 'uploaded_slide' in st.session_state:
    slide = st.session_state.uploaded_slide
    st.header("Análisis de Normalización")


    # Usar cache para la región pequeña inicial
    @st.cache_data
    def get_initial_region(_slide):
        region = _slide.read_region((10000, 4000), 0, (5000, 5000))
        region_RGB = region.convert('RGB')
        return np.array(region_RGB)


    region_small_np = get_initial_region(slide)
    plt.axis('off')
    st.image(region_small_np)

    # Sidebar controls
    st.sidebar.header("Ajustes de TRRef")
    trref_values = {
        "00": st.sidebar.slider("Coeficiente Colágeno (Colorante 1 a 1)", 0.0, 1.0, 0.7, 0.01),
        "01": st.sidebar.slider("Coeficiente Colorante 1 a Otro (Colorante 1 a 2)", 0.0, 1.0, 0.1, 0.01),
        "10": st.sidebar.slider("Coeficiente Otro a Colágeno (Colorante 2 a 1)", 0.0, 1.0, 0.15, 0.01),
        "11": st.sidebar.slider("Coeficiente Otro (Colorante 2 a 2)", 0.0, 1.0, 0.9, 0.01),
        "20": st.sidebar.slider("Coeficiente Tercer Colorante a Colágeno (Colorante 3 a 1)", 0.0, 1.0, 0.2, 0.01),
        "21": st.sidebar.slider("Coeficiente Tercer Colorante a Otro (Colorante 3 a 2)", 0.0, 1.0, 0.8, 0.01)
    }

    TRRef = np.array([[trref_values["00"], trref_values["01"]],
                      [trref_values["10"], trref_values["11"]],
                      [trref_values["20"], trref_values["21"]]])

    maxCRef = np.array([
        st.sidebar.slider("Concentración máxima de colágeno (maxCRef[0])", 0.0, 5.0, 2.0, 0.1),
        st.sidebar.slider("Concentración máxima de otro colorante (maxCRef[1])", 0.0, 5.0, 1.0, 0.1)
    ])

    # Usar el cache para la normalización
    norm_img, collagen_img, other_img = cached_norm_Masson(region_small_np, TRRef, maxCRef, 240, 1, 0.20)

    # Mostrar imágenes en columnas
    col1, col2 = st.columns(2)
    with col1:
        st.image(region_small_np, caption="Original Image", use_column_width=True)
        st.image(collagen_img, caption="Collagen Image", use_column_width=True)
    with col2:
        st.image(norm_img, caption="Normalized Image", use_column_width=True)
        st.image(other_img, caption="Other Things Image", use_column_width=True)

    # Procesar máscaras usando cache
    min_val = st.sidebar.slider("Valor mínimo", 0, 255, 50)
    max_val = st.sidebar.slider("Valor máximo", 0, 255, 150)
    normalized_img, mask, result = process_image_masks(collagen_img, min_val, max_val)

    st.image(normalized_img, caption="Imagen Normalizada", use_column_width=True)
    st.image(mask, caption=f"Máscara (valores entre {min_val} y {max_val})", use_column_width=True)
    st.image(result, caption="Resultado con máscara aplicada", use_column_width=True)

    # Selección de punto en la imagen
    st.title("Selecciona un punto en la imagen")
    image_np, best_level, level_dimensions = get_slide_visualization(slide)

    # Canvas configuration
    canvas_width, canvas_height = 704, 200
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=Image.fromarray(image_np),
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="rect",
        point_display_radius=20,
        key="canvas",
    )

    if canvas_result.json_data is not None:
        for shape in canvas_result.json_data["objects"]:
            if shape["type"] == "rect":
                # Calcular coordenadas
                left, top = int(shape["left"]), int(shape["top"])
                width, height = int(shape["width"]), int(shape["height"])

                scale_x = level_dimensions[0] / canvas_width
                scale_y = level_dimensions[1] / canvas_height

                top_left_x = int(left * scale_x)
                top_left_y = int(top * scale_y)
                bottom_right_x = int((left + width) * scale_x)
                bottom_right_y = int((top + height) * scale_y)

                scale_factor = slide.level_downsamples[best_level]
                top_left_x_0 = int(top_left_x * scale_factor)
                top_left_y_0 = int(top_left_y * scale_factor)
                bottom_right_x_0 = int(bottom_right_x * scale_factor)
                bottom_right_y_0 = int(bottom_right_y * scale_factor)

                region_width = bottom_right_x_0 - top_left_x_0
                region_height = bottom_right_y_0 - top_left_y_0

                # Obtener y procesar región usando cache
                region_image = get_region_image(slide, top_left_x_0, top_left_y_0, region_width, region_height)
                region_image_np = np.array(region_image)

                # Mostrar coordenadas
                st.write(f"Coordenadas en nivel {best_level}:")
                st.write(f"Esquina superior izquierda: ({top_left_x}, {top_left_y})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x}, {bottom_right_y})")
                st.write(f"Coordenadas en nivel 0:")
                st.write(f"Esquina superior izquierda: ({top_left_x_0}, {top_left_y_0})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x_0}, {bottom_right_y_0})")
                st.image(region_image, caption=f"Región seleccionada en nivel 0")

                # Normalizar región seleccionada usando cache
                norm_img, collagen_img, other_img = cached_norm_Masson(region_image_np, TRRef, maxCRef, 240, 1, 0.20)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(region_image_np, caption="Original Image", use_column_width=True)
                    st.image(collagen_img, caption="Collagen Image", use_column_width=True)
                with col2:
                    st.image(norm_img, caption="Normalized Image", use_column_width=True)
                    st.image(other_img, caption="Other Things Image", use_column_width=True)

else:
    st.markdown("---")
    st.header("Por favor, selecciona una imágen para poder continuar")
    st.markdown("---")