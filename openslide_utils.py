import numpy as np
import os
import openslide
from dotenv import load_dotenv

load_dotenv()  # Cargar variables de entorno desde .env
OPENSLIDE_PATH = os.getenv("OPENSLIDE_PATH")
from ctypes import cdll

dll_path = os.path.join(OPENSLIDE_PATH, 'libopenslide-1.dll')
cdll.LoadLibrary(dll_path)
def load_slide_image(slide_path):

    """
    Loads an image from an OpenSlide file and converts it to a NumPy array.

    Parameters:
        slide_path (str): The file path to the OpenSlide image.

    Returns:
        numpy.ndarray: A NumPy array representation of the slide's thumbnail image.
    """
    slide = openslide.open_slide(slide_path)
    slide_thumbnail = slide.get_thumbnail((1000, 1000))
    slide_array = np.array(slide_thumbnail)
    img_array = np.array(slide_array)  # Convertir a array de NumPy
    return img_array


# Función para determinar el mejor nivel de visualización basado en el tamaño máximo
def get_best_level(slide):
    """
       Determines the best level of the slide for visualization based on a maximum size criterion.

       Parameters:
           slide (openslide.OpenSlide): The OpenSlide object representing the slide.

       Returns:
           int: The index of the best level for visualization.
       """
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
    """
       Loads a specific region of the slide image at the given level.

       Parameters:
           slide (openslide.OpenSlide): The OpenSlide object representing the slide.
           level (int): The level of the slide to load the region from.

       Returns:
           tuple: A tuple containing:
               - numpy.ndarray: The region of the slide as a NumPy array in RGB format.
               - tuple: The dimensions of the region loaded (width, height).
       """
    # Obtener las dimensiones del nivel seleccionado
    level_size = slide.level_dimensions[level]

    # Leer toda la región en ese nivel
    region = slide.read_region((0, 0), level, level_size)

    # Convertir a formato RGB
    region_rgb = region.convert('RGB')
    region_np = np.array(region_rgb)

    return region_np, level_size