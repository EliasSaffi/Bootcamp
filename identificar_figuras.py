#        >>> Elias Kalil Maniá Saffi - 4200987 <<<

import cv2
import numpy as np

# Endereço da imagem
image = cv2.imread('C:/Users/Elias/Downloads/T1_formas_por_cor/formas.JPG')

# Redimensionar a imagem
image = cv2.resize(image, (500, 500))

# Escala de cinza para suavização
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# etecção das bordas
edges = cv2.Canny(blurred, 50, 150)

# Detectar contorno
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identificar formas por numero de lados
def identify_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangulo"
    elif sides == 4:
        # Quadrado ou retangulo (ou outras com mais lados)
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Quadrado"
        else:
            return "Retangulo"
    elif sides == 5:
        return "Pentagono"
    elif sides == 6:
        return "Hexagono"
    else:
        return "Circulo"

# Checar contorno
for contour in contours:
    # Verificar contorno para chegar a forma
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Encontrar forma
    shape = identify_shape(approx)
    
    # Escrever nome no centro da forma
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    # Somente nome da forma
    cv2.putText(image, shape, (cX - 50, cY), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Identificar imagem pela forma 
cv2.imshow("Identificação de Formas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
