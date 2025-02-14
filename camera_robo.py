import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv8
model = YOLO('/home/robo/yolov8n.pt')  # Caminho para seu arquivo .pt local ou um modelo pré-treinado

# Configurar a câmera
cap = cv2.VideoCapture(2)  # Tenta abrir a câmera (0, 1, ou 2 se necessário) sendo 0 que serve para abrir uma camera USB

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

    # Captura frame por frame
ret, frame = cap.read()
if not ret:
    print("Falha ao capturar imagem.")
         

# Realizar a detecção de objetos
results = model(frame)  # Passa o frame para o modelo

# Acessar o primeiro item da lista de resultados e chamar o método show()
results[0].show()  # Exibe a imagem com as deteções (acessando o primeiro resultado)

# Exibir a imagem com as deteções
cv2.imshow('Detecção de Jogadores', frame)

# Libere a câmera e feche as janelas
cap.release()
cv2.destroyAllWindows()

#comando para abrir o video: mpv av://v4l2:/dev/video0
