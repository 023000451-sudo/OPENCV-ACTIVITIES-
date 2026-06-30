# Python program to illustrate foreground extraction using GrabCut algorithm

# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 

# --- CONFIGURACIÓN DE RUTAS ---
# Localiza la carpeta raíz del proyecto subiendo un nivel desde 'src'
DIR_DEL_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(DIR_DEL_SCRIPT)

# Ruta de entrada (data) y de salida (out)
folder_original = os.path.join(RAIZ_PROYECTO, 'data')
folder_procesada = os.path.join(RAIZ_PROYECTO, 'out')

# Asegurar que la carpeta 'out' exista
if not os.path.exists(folder_procesada):
    os.makedirs(folder_procesada)

# --- CAMBIO AQUÍ: Ahora busca 'image.png' con tu extensión correcta ---
ruta_entrada = os.path.join(folder_original, 'image.png')

# Verificación de seguridad por si no encuentra la imagen
if not os.path.exists(ruta_entrada):
    print(f"\n ERROR: No se encontró 'image.png' en la ruta:")
    print(f" {ruta_entrada}\n")
    exit()

# Cargar la imagen original
image = cv2.imread(ruta_entrada)
 
# create a simple mask image similar
# to the loaded image, with the 
# shape and return type
mask = np.zeros(image.shape[:2], np.uint8)
 
# specify the background and foreground model
# using numpy the array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
 
# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
# NOTA: Asegúrate de que tu objeto esté dentro de este cuadro (X, Y, Ancho, Alto)
# --- RECUADRO AJUSTADO MANUALMENTE PARA EL ROSTRO ---
altura, ancho = image.shape[:2]

# Definimos un cuadro centrado que cubra la cabeza del hombre
x_inicio = int(ancho * 0.15)
y_inicio = int(altura * 0.05)
ancho_box = int(ancho * 0.70)
alto_box = int(altura * 0.90)

rectangle = (x_inicio, y_inicio, ancho_box, alto_box)
# -----------------------------------------------------
 
# apply the grabcut algorithm with appropriate
# values as parameters, number of iterations = 3 
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used 
cv2.grabCut(image, mask, rectangle,   
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)
 
# In the new mask image, pixels will 
# be marked with four flags 
# four flags denote the background / foreground 
# mask is changed, all the 0 and 2 pixels 
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
 
# The final mask is multiplied with 
# the input image to give the segmented image.
image_segmented = image * mask2[:, :, np.newaxis]
 
# --- Guardar AMBAS imágenes en la carpeta 'out' ---
# 1. Guarda la copia de la imagen original en 'out' (la guardamos como .png también)
cv2.imwrite(os.path.join(folder_procesada, 'imagen_original.png'), image)

# 2. Guarda la imagen procesada por GrabCut en 'out'
cv2.imwrite(os.path.join(folder_procesada, 'resultado_grabcut.png'), image_segmented)

print("¡Hecho! Se guardaron 'imagen_original.png' y 'resultado_grabcut.png' en la carpeta /out")
# ---------------------------------------------------------

# output segmented image with colorbar
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display the segmented image
plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()