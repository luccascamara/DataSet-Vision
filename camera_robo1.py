import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO uma única vez
model = YOLO('/home/luccascamara/yolov8n.pt')

# Abrir a câmera apenas uma vez
cap = cv2.VideoCapture(2)  

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

print("Câmera iniciada! Pressione 'q' para sair.")

# Criar janela única antes do loop para evitar múltiplas instâncias
cv2.namedWindow("Detecção ao Vivo - YOLOv8", cv2.WINDOW_NORMAL)

try:
    while cap.isOpened():  # Garantir que apenas uma câmera esteja ativa
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break

        # Realizar detecção apenas uma vez por frame
        results = model.predict(frame, verbose=False)  

        # Desenhar as caixas delimitadoras na imagem
        if results and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                conf = box.conf[0].item()  
                label = results[0].names[int(box.cls[0])]  

                # Desenhar retângulo ao redor do objeto detectado
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Exibir apenas UMA janela de detecção
        cv2.imshow("Detecção ao Vivo - YOLOv8", frame)

        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExecução interrompida pelo usuário.")

except Exception as e:
    print(f"\nErro inesperado: {e}")

finally:
    print("\nFinalizando...")
    cap.release()  # Liberar a câmera corretamente
    cv2.destroyAllWindows()  # Fechar todas as janelas corretamente
    print("Câmera desligada e janela fechada com sucesso!")
