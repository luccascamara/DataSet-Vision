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

#ola chatgpt, estou com codigo pronto no meu computador linux que identifica pessoas e objetos. porem, ele esta tirando apenas uma foto com a webcam e parando de executar o codigo. preciso que voce mude o codigo que eu vou lhe enviar para que fique um video constante e que seja um codigo que nao abra mil janelas de video e trave o pc, isso acontece gracas ao while true

#comando para abrir o video: mpv av://v4l2:/dev/video0
