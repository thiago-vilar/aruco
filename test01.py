import cv2
from object_detector import HomogeneousBgDetector
import numpy as np

# Carrega o detector Aruco
parameters = cv2.aruco.DetectorParameters() 

# Carrega o dicionário Aruco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50) 

# Carrega o detector de fundo homogêneo
detector = HomogeneousBgDetector()

# Carrega a imagem
img = cv2.imread("phone_aruco.jpg")
if img is None:
    raise FileNotFoundError("A imagem não foi encontrada.")

# Detecta marcadores Aruco na imagem
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Se marcadores foram encontrados, prossegue com o desenho e medições
if ids is not None and len(corners) > 0:
    # Desenha os polígonos dos marcadores Aruco
    int_corners = [np.int0(corner) for corner in corners]
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # Calcula o perímetro do primeiro marcador Aruco detectado
    aruco_perimeter = cv2.arcLength(corners[0], True)
    pixel_cm_ratio = aruco_perimeter / 20

    # Detecta objetos usando o detector de fundo homogêneo
    contours = detector.detect_objects(img)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Desenha o retângulo e marca o centro
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 255), 2)
        cv2.putText(img, "Width {:.1f} cm".format(w / pixel_cm_ratio), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 222, 0), 2)
        cv2.putText(img, "Height {:.1f} cm".format(h / pixel_cm_ratio), (int(x - 100), int(y + 20)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 222, 0), 2)

        print(box)
        print(x, y, w, h, angle)

# imagem processada
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
