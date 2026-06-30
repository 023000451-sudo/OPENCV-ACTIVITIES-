import cv2
import numpy as np
import os

# 1. Asegurar que ambas carpetas existan en el proyecto
folder_original = 'data'
folder_procesada = 'out'

for folder in [folder_original, folder_procesada]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame and convert to float
_, img = cap.read()
averageValue1 = np.float32(img)

while True:
    # Capture next frame
    _, img = cap.read()
    
    # Update background model
    cv2.accumulateWeighted(img, averageValue1, 0.02)
    
    # Convert back to 8-bit for display
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)

    # Show both original and background model
    cv2.imshow('Original Frame', img)
    cv2.imshow('Background (Running Average)', resultingFrames1)
    
    # Exit on Esc key
    if cv2.waitKey(30) & 0xFF == 27:
        # --- NUEVO: Guardar ambas imágenes al salir ---
        
        # Guardar la imagen original en la carpeta 'data'
        path_original = os.path.join(folder_original, 'imagen_original.png')
        cv2.imwrite(path_original, img)
        
        # Guardar la imagen procesada en la carpeta 'out'
        path_procesada = os.path.join(folder_procesada, 'resultado_fondo.png')
        cv2.imwrite(path_procesada, resultingFrames1)
        
        print("¡Imágenes guardadas correctamente!")
        print(f"Original en: {path_original}")
        print(f"Procesada en: {path_procesada}")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()