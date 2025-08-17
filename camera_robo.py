import cv2
from ultralytics import YOLO

model = YOLO('/home/robo/yolov8n.pt')  

cap = cv2.VideoCapture(2)  

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Falha ao capturar imagem.")
         
results = model(frame)  

results[0].show()  

cv2.imshow('Detecção de Jogadores', frame)

cap.release()
cv2.destroyAllWindows()
