import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicia captura de vídeo
cap = cv2.VideoCapture(0)

# Carrega a imagem (meme)
meme = cv2.imread("hamster-language.png")  # coloque o caminho da sua imagem
if meme is None:
    raise FileNotFoundError("A imagem 'meme.png' não foi encontrada.")

# Redimensiona o meme para ficar pequeno
meme = cv2.resize(meme, (200, 200))  # largura, altura

with mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape
                # Pega coordenadas dos lábios superior e inferior
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                dist = abs(top_lip.y - bottom_lip.y)

                # Se a boca abrir mais que um certo limite → mostra meme
                if dist > 0.03:
                    h_img, w_img, _ = meme.shape

                    # Posição canto superior esquerdo
                    y_offset, x_offset = 20, 20

                    # Sobrepõe a imagem (sem ultrapassar bordas)
                    frame[y_offset:y_offset + h_img, x_offset:x_offset + w_img] = meme

#             mp_drawing.draw_landmarks(
#                   frame,
#                   face_landmarks,
#                    mp_face.FACEMESH_CONTOURS,
#                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
#               )

        cv2.imshow("Detector de Expressões", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

cap.release()
cv2.destroyAllWindows()
