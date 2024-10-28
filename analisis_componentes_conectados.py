import streamlit as st
import numpy as np
import os
import cv2
from openslide_utils import get_best_level, load_slide_image, load_slide_region

# Función para cargar la imagen de OpenSlide y convertirla a un array de NumPy


if 'uploaded_slide' in st.session_state:
    slide = st.session_state.uploaded_slide

    np_image, _ = load_slide_region(slide, get_best_level(slide))

    # Subtítulo para procesamiento de la imagen
    st.header("Procesamiento de la Imagen")

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para binarizar la imagen
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Análisis de componentes conectados
    analysis = cv2.connectedComponentsWithStats(threshold, 8, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    area_threshold_slider = st.slider('Valor minimo de area a considerar', min_value=1,
                                      max_value=2000)
    # Inicializar una nueva imagen para almacenar los componentes filtrados
    output = np.zeros(gray.shape, dtype="uint8")

    # Iterar sobre los componentes encontrados
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        if area > area_threshold_slider:  # Filtrar por tamaño mínimo de área

            # Crear una copia de la imagen original
            new_img = np_image.copy()

            # Extraer las coordenadas del cuadro delimitador
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]

            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1 + w, y1 + h)
            (X, Y) = centroid[i]

            # Dibujar cuadros delimitadores
            # Bounding boxes for each component
            cv2.rectangle(new_img, pt1, pt2,
                          (0, 255, 0), 3)
            cv2.circle(new_img, (int(X),
                                 int(Y)),
                       4, (0, 0, 255), -1)

            # Create a new array to show individual component
            component = np.zeros(gray.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255

            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component, componentMask)
            output = cv2.bitwise_or(output, componentMask)

            # Mostrar la imagen con cuadros delimitadores
            st.image(new_img, caption="Imagen con cuadros delimitadores", use_column_width=True)

    # Inicializar lista para almacenar los datos de cada componente
    component_data = []

    # Iterar sobre cada componente (excepto el fondo)
    for i in range(1, totalLabels):
        # Crear una máscara binaria para el componente i
        component_mask = (label_ids == i).astype("uint8") * 255  # Asegurar que la máscara esté en el rango 0-255

        # Calcular el área del componente
        area = cv2.countNonZero(component_mask)

        # Filtrar los componentes con un área mayor a 1000
        if area > area_threshold_slider:
            # Encontrar los contornos del componente
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calcular el área del contorno
            contour_area = cv2.contourArea(contours[0])

            # Calcular el porcentaje del área de la máscara respecto al área del contorno
            percentage = (area / contour_area) * 100

            # Crear una imagen en blanco para dibujar el contorno
            contour_img = np.zeros_like(component_mask)

            # Dibujar el contorno en la imagen en blanco
            cv2.drawContours(contour_img, contours, -1, 255, thickness=2)

            # Guardar los datos del componente
            component_data.append((component_mask, contour_img, area, contour_area, percentage))

    # Mostrar los componentes en Streamlit
    total_percentage_of_image = None
    # Separador visual

    if component_data:
        st.write(f"Se encontraron {len(component_data)} componentes con área mayor a {area_threshold_slider}.")
        st.markdown("---")
        # Inicializar variables para acumular áreas
        sum_area = 0
        sum_contour = 0

        for i, (mask, contour_img, mask_area, contour_area, percentage) in enumerate(component_data):
            st.write(f"### Componente {i + 1}")

            # Crear columnas para mostrar imágenes lado a lado
            col1, col2 = st.columns(2)

            # Mostrar la máscara en la primera columna
            with col1:
                st.image(mask, caption=f"Máscara (Área: {mask_area})", use_column_width=True, channels="GRAY")

            # Mostrar el contorno en la segunda columna
            with col2:
                st.image(contour_img, caption=f"Contorno (Área: {contour_area}, Porcentaje: {percentage:.2f}%)",
                         use_column_width=True, channels="GRAY")

            # Acumular las áreas
            sum_area += mask_area
            sum_contour += contour_area

        # Calcular y mostrar el porcentaje total
        total_percentage = sum_area / sum_contour
        total_percentage_of_image = total_percentage
        st.write(f"### Porcentaje total de area cubierta: {total_percentage:.2f}%")
        st.write('Mientras más cerca de 1, más fibrosis hay')
    else:
        st.write(f"No se encontraron componentes con área mayor a {area_threshold_slider}.")

    # Separador visual
    st.markdown("---")

    # Subtítulo para las máscaras y contornos coloreados
    st.write(f"### Máscaras y Contornos Coloreados")

    # Crear una imagen en escala de grises para dibujar las máscaras
    masks_image = np.zeros_like(label_ids, dtype=np.uint8)

    # Crear una imagen en blanco para dibujar los contornos en color
    final_image = np.zeros((label_ids.shape[0], label_ids.shape[1], 3), dtype=np.uint8)

    # Definir el color amarillo (en formato BGR, que es el utilizado por OpenCV)
    yellow_color = (0, 255, 255)  # Azul = 0, Verde = 255, Rojo = 255

    # Iterar sobre cada componente conectado (excepto el fondo)
    for i in range(1, totalLabels):
        # Crear una máscara binaria para el componente i
        component_mask = np.uint8(label_ids == i)

        # Calcular el área del componente
        area = cv2.countNonZero(component_mask)

        # Filtrar los componentes con un área mayor a 1000
        if area > area_threshold_slider:
            # Añadir la máscara al fondo gris (intensidad 255 para que la máscara sea blanca)
            masks_image = cv2.add(masks_image, component_mask * 255)

            # Encontrar los contornos del componente
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dibujar el contorno en la imagen en color con un color diferente
            cv2.drawContours(final_image, contours, -1, yellow_color, thickness=2)

    # Convertir la imagen de máscaras a color (escala de grises a RGB)
    masks_image_rgb = cv2.cvtColor(masks_image, cv2.COLOR_GRAY2RGB)

    # Combinar la imagen de las máscaras con los contornos de color
    combined_image = cv2.addWeighted(masks_image_rgb, 0.7, final_image, 0.7, 0)

    # Mostrar la imagen combinada de máscaras y contornos en Streamlit
    st.image(combined_image, caption="Máscaras y contornos coloreados", channels="RGB", use_column_width=True)

    # Separador visual
    st.markdown("---")

else:
    st.markdown("---")
    st.header("Por favor, selecciona una imágen para poder continuar")
    st.markdown("---")
