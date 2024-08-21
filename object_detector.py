import cv2

class HomogeneousBgDetector:
    def __init__(self):
        # Inicialização, se necessário, pode ser expandida aqui
        pass

    def detect_objects(self, frame):
        # Convertendo o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicando limiarização adaptativa para segmentar a imagem
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        # Encontrando contornos na imagem binarizada
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtrando contornos por área para identificar objetos maiores
        objects_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
        
        return objects_contours
