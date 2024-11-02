import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib import pyplot as plt
import os
import cv2
from openslide_utils import load_slide_image, load_slide_region, get_best_level
from normalization_utils import norm_Masson_modified as norm_Masson

@st.cache
def get_region_image(slide, top_left_x_0, top_left_y_0, region_width, region_height):
    return slide.read_region((top_left_x_0, top_left_y_0), 0, (region_width, region_height)).convert('RGB')


if 'uploaded_slide' in st.session_state:
    slide = st.session_state.uploaded_slide
    # Subtítulo para las máscaras y contornos coloreados
    st.header("Análisis de Normalización")

    region_small = slide.read_region((10000, 4000), 0, (5000, 5000))
    region_small_RGB = region_small.convert('RGB')
    region_small_np = np.array(region_small_RGB)

    plt.axis('off')
    st.image(region_small_np)
    # Crear seis sliders para los valores de la matriz TRRef
    st.sidebar.header("Ajustes de TRRef")
    trref_00 = st.sidebar.slider("Coeficiente Colágeno (Colorante 1 a 1)", min_value=0.0, max_value=1.0, value=0.7,
                                 step=0.01)
    trref_01 = st.sidebar.slider("Coeficiente Colorante 1 a Otro (Colorante 1 a 2)", min_value=0.0, max_value=1.0,
                                 value=0.1, step=0.01)
    trref_10 = st.sidebar.slider("Coeficiente Otro a Colágeno (Colorante 2 a 1)", min_value=0.0, max_value=1.0,
                                 value=0.15, step=0.01)
    trref_11 = st.sidebar.slider("Coeficiente Otro (Colorante 2 a 2)", min_value=0.0, max_value=1.0, value=0.9,
                                 step=0.01)
    trref_20 = st.sidebar.slider("Coeficiente Tercer Colorante a Colágeno (Colorante 3 a 1)", min_value=0.0,
                                 max_value=1.0, value=0.2, step=0.01)
    trref_21 = st.sidebar.slider("Coeficiente Tercer Colorante a Otro (Colorante 3 a 2)", min_value=0.0, max_value=1.0,
                                 value=0.8, step=0.01)

    # Crear la matriz TRRef con los valores seleccionados en los sliders
    TRRef = np.array([[trref_00, trref_01],
                      [trref_10, trref_11],
                      [trref_20, trref_21]])

    maxCRef_0 = st.sidebar.slider("Concentración máxima de colágeno (maxCRef[0])", min_value=0.0, max_value=5.0,
                                  value=2.0, step=0.1)
    maxCRef_1 = st.sidebar.slider("Concentración máxima de otro colorante (maxCRef[1])", min_value=0.0, max_value=5.0,
                                  value=1.0, step=0.1)

    # Crear el array maxCRef con los valores seleccionados en los sliders
    maxCRef = np.array([maxCRef_0, maxCRef_1])

    # Realizar la normalización con los valores ajustados
    norm_img, collagen_img, other_img = norm_Masson(region_small_np, TRRef, maxCRef, Io=240, alpha=1, beta=0.20)

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Mostrar imágenes en la primera columna
    with col1:
        st.image(region_small_np, caption="Original Image", use_column_width=True)
        st.image(collagen_img, caption="Collagen Image", use_column_width=True)

    # Mostrar imágenes en la segunda columna
    with col2:
        st.image(norm_img, caption="Normalized Image", use_column_width=True)
        st.image(other_img, caption="Other Things Image", use_column_width=True)

    # Convertir la imagen a escala de grises si tiene múltiples canales

    if len(collagen_img.shape) == 3:
        gray_img = cv2.cvtColor(collagen_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = collagen_img

    # Normalizar la imagen para mejorar el contraste
    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Mostrar la imagen normalizada
    st.image(normalized_img, caption="Imagen Normalizada", use_column_width=True)

    # Slider para ajustar un umbral mínimo y máximo
    st.sidebar.title("Ajustes de Umbral")
    min_val = st.sidebar.slider("Valor mínimo", min_value=0, max_value=255, value=50)
    max_val = st.sidebar.slider("Valor máximo", min_value=0, max_value=255, value=150)

    # Aplicar una máscara basada en el rango de grises
    mask = cv2.inRange(normalized_img, min_val, max_val)

    # Convertir la imagen a BGR si es necesario para aplicar la máscara correctamente
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Aplicar la máscara a la imagen
    result = cv2.bitwise_and(img_bgr, region_small_np, mask=mask)

    # Mostrar la máscara y el resultado
    st.image(mask, caption=f"Máscara (valores entre {min_val} y {max_val})", use_column_width=True)
    st.image(result, caption="Resultado con máscara aplicada", use_column_width=True)

    # Título de la app
    st.title("Selecciona un punto en la imagen")

    # Determinar el mejor nivel de visualización
    best_level = get_best_level(slide)

    # Cargar la imagen completa del nivel adecuado
    image_np, image_size = load_slide_region(slide, best_level)

    # Mostrar la imagen cargada en Streamlit
    # st.image(image_np, caption=f"Imagen de (Nivel {best_level}, Tamaño {image_size})", use_column_width=True)

    # Escalar la imagen si es muy grande (opcional, dependiendo del tamaño de la imagen)
    image_height, image_width = image_np.shape[:2]
    max_dimension = 2000  # Limitar la dimensión máxima a 2000 px
    scale_factor = min(max_dimension / image_height, max_dimension / image_width)

    # Nivel óptimo para visualización
    best_level = slide.get_best_level_for_downsample(16)
    level_dimensions = slide.level_dimensions[best_level]

    # Escalar la imagen y convertir a array
    image = slide.read_region((0, 0), best_level, level_dimensions)
    image_np = np.array(image.convert("RGB"))

    # Tamaño fijo del canvas
    canvas_width, canvas_height = 704, 200

    # Crear lienzo interactivo
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

    # Verificar si se ha seleccionado un punto
    if canvas_result.json_data is not None:
        for shape in canvas_result.json_data["objects"]:
            if shape["type"] == "rect":
                # Obtener las coordenadas y dimensiones del rectángulo en el canvas
                left = int(shape["left"])
                top = int(shape["top"])
                width = int(shape["width"])
                height = int(shape["height"])

                # Calcular el factor de escala entre el canvas y las dimensiones reales
                scale_x = level_dimensions[0] / canvas_width
                scale_y = level_dimensions[1] / canvas_height

                # Convertir las coordenadas de los cuatro vértices a `level_dimensions`
                top_left_x = int(left * scale_x)
                top_left_y = int(top * scale_y)
                bottom_right_x = int((left + width) * scale_x)
                bottom_right_y = int((top + height) * scale_y)

                # Convertir las coordenadas de `level_dimensions` a nivel 0
                scale_factor = slide.level_downsamples[best_level]
                top_left_x_0 = int(top_left_x * scale_factor)
                top_left_y_0 = int(top_left_y * scale_factor)
                bottom_right_x_0 = int(bottom_right_x * scale_factor)
                bottom_right_y_0 = int(bottom_right_y * scale_factor)

                # Calcular el tamaño de la región en nivel 0
                region_width = bottom_right_x_0 - top_left_x_0
                region_height = bottom_right_y_0 - top_left_y_0

                # Leer la región seleccionada en nivel 0
                region_image = get_region_image(slide, top_left_x_0, top_left_y_0, region_width, region_height)
                region_image_np = np.array(region_image)

                # Mostrar resultados
                st.write(f"Coordenadas en nivel {best_level}:")
                st.write(f"Esquina superior izquierda: ({top_left_x}, {top_left_y})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x}, {bottom_right_y})")
                st.write(f"Coordenadas en nivel 0:")
                st.write(f"Esquina superior izquierda: ({top_left_x_0}, {top_left_y_0})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x_0}, {bottom_right_y_0})")
                st.image(region_image, caption=f"Región seleccionada en nivel 0")

                norm_img, collagen_img, other_img = norm_Masson(region_image_np, TRRef,maxCRef, Io=240, alpha=1, beta=0.20)
                # Crear dos columnas
                col1, col2 = st.columns(2)

                # Mostrar imágenes en la primera columna
                with col1:
                    st.image(region_image_np, caption="Original Image", use_column_width=True)
                    st.image(collagen_img, caption="Collagen Image", use_column_width=True)

                # Mostrar imágenes en la segunda columna
                with col2:
                    st.image(norm_img, caption="Normalized Image", use_column_width=True)
                    st.image(other_img, caption="Other Things Image", use_column_width=True)


else:
    st.markdown("---")
    st.header("Por favor, selecciona una imágen para poder continuar")
    st.markdown("---")