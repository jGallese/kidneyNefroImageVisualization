# The path can also be read from a config file, etc.
import tempfile

OPENSLIDE_PATH = r'C:\Dev\openslide-bin-4.0.0.4-windows-x64\bin'
import os

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import open_slide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
# Añadir CSS para ajustar márgenes y bordes
# Configura la página para que ocupe todo el ancho disponible


# Ajuste de la función norm_Masson para aceptar la matriz TRRef como parámetro
def norm_Masson(img, TRRef, Io=240, alpha=1, beta=0.15):
    # reference maximum stain concentrations for Masson
    maxCRef = np.array([2.0, 2])

    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log10((img.astype(float) + 1) / Io)

    ############ Step 2: Remove data with OD intensity less than β ############
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Remove transparent pixels

    ############# Step 3: Calculate SVD on the OD tuples ######################
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    ######## Step 4: Create plane from the SVD directions with two largest values ######
    That = ODhat.dot(eigvecs[:, 1:3])  # Project on the plane

    ############### Step 5: Normalize to unit length ###########
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        TR = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        TR = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(TR, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    Inorm = np.multiply(Io, np.exp(-TRRef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating components (collagen and other stains)
    Collagen = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    Collagen[Collagen > 255] = 254
    Collagen = np.reshape(Collagen.T, (h, w, 3)).astype(np.uint8)

    Other = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    Other[Other > 255] = 254
    Other = np.reshape(Other.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, Collagen, Other)

# Función para cargar la imagen de OpenSlide y convertirla a un array de NumPy
def load_slide_image(slide_path):
    slide = openslide.open_slide(slide_path)
    slide_thumbnail = slide.get_thumbnail((1000, 1000))
    slide_array = np.array(slide_thumbnail)
    img_array = np.array(slide_array)  # Convertir a array de NumPy
    return img_array


# Función para determinar el mejor nivel de visualización basado en el tamaño máximo
def get_best_level(slide):
    max_size = 2000  # Tamaño máximo deseado para visualización (ajustable)

    # Iterar sobre los niveles de la imagen
    for level in range(slide.level_count):
        # Obtener las dimensiones de ese nivel
        level_size = slide.level_dimensions[level]
        if max(level_size) <= max_size:
            return level  # Retornar el nivel más adecuado
    return slide.level_count - 1  # Si ninguno cumple, usar el nivel más bajo disponible


# Función para cargar una región específica de la imagen en el nivel adecuado
def load_slide_region(slide, level):
    # Obtener las dimensiones del nivel seleccionado
    level_size = slide.level_dimensions[level]

    # Leer toda la región en ese nivel
    region = slide.read_region((0, 0), level, level_size)

    # Convertir a formato RGB
    region_rgb = region.convert('RGB')
    region_np = np.array(region_rgb)

    return region_np, level_size

# Crear un título principal
st.title("Análisis de Imágenes de Microscopía con OpenSlide y OpenCV")
tab1, tab2, tab3 = st.tabs(["Carga de Imágen", "Análisis de Componentes Conectados", "Análisis de Normalización"])


with tab1:
    uploaded_file = st.file_uploader("Elegí un archivo", type=["tif", "tiff", "cvs", ])
    # Subtítulo para la selección de imágenes
    st.header("Selección de Imagen")

    if uploaded_file is not None:
        # Crea un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            slide_path = temp_file.name  # Guarda la ruta del archivo temporal

        # Intentar cargar la imagen de OpenSlide
        try:
            slide = open_slide(slide_path)  # Cargar la imagen
            img = load_slide_image(slide_path)
            np_image, _ = load_slide_region(slide, get_best_level(slide))

            # Mostrar la imagen cargada en Streamlit
            st.image(np_image, caption="Imagen cargada", use_column_width=True)

            # Cierra la imagen de OpenSlide y elimina el archivo temporal
        except Exception as e:
            st.error(f"Ocurrió un error al cargar la imagen: {e}")

with tab2:

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

with tab3:
    # Subtítulo para las máscaras y contornos coloreados
    st.header("Análisis de Normalización")


    region_small = slide.read_region((1000,2000), 0, (2000,2000))
    region_small_RGB = region_small.convert('RGB')
    region_small_np = np.array(region_small_RGB)

    plt.axis('off')
    st.image(region_small_np)
    # Crear seis sliders para los valores de la matriz TRRef
    st.sidebar.header("Ajustes de TRRef")

    trref_00 = st.sidebar.slider("TRRef[0,0]", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    trref_01 = st.sidebar.slider("TRRef[0,1]", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    trref_10 = st.sidebar.slider("TRRef[1,0]", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    trref_11 = st.sidebar.slider("TRRef[1,1]", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    trref_20 = st.sidebar.slider("TRRef[2,0]", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    trref_21 = st.sidebar.slider("TRRef[2,1]", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    # Crear la matriz TRRef con los valores seleccionados en los sliders
    TRRef = np.array([[trref_00, trref_01],
                      [trref_10, trref_11],
                      [trref_20, trref_21]])

    # Realizar la normalización con los valores ajustados
    norm_img, collagen_img, other_img = norm_Masson(region_small_np, TRRef, Io=240, alpha=1, beta=0.20)

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
                region_image = slide.read_region(
                    (top_left_x_0, top_left_y_0), 0, (region_width, region_height)
                )

                # Mostrar resultados
                st.write(f"Coordenadas en nivel {best_level}:")
                st.write(f"Esquina superior izquierda: ({top_left_x}, {top_left_y})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x}, {bottom_right_y})")
                st.write(f"Coordenadas en nivel 0:")
                st.write(f"Esquina superior izquierda: ({top_left_x_0}, {top_left_y_0})")
                st.write(f"Esquina inferior derecha: ({bottom_right_x_0}, {bottom_right_y_0})")
                st.image(region_image, caption=f"Región seleccionada en nivel 0")
