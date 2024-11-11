import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega os arquivos de áudio
audio_file = pygame.mixer.Sound("songs/effect.mp3")  # Som para boca aberta
audio_file2 = pygame.mixer.Sound("songs/acorde.mp3")  # Som para olhos fechados

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função para calcular o EAR (Eye Aspect Ratio)
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * np.linalg.norm(face_esq[4] - face_esq[5]))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * np.linalg.norm(face_dir[4] - face_dir[5]))
        return (ear_esq + ear_dir) / 2
    except:
        return 0.0

# Função para calcular o MAR (Mouth Aspect Ratio)
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]
        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * np.linalg.norm(face_boca[6] - face_boca[7]))
        return mar
    except:
        return 0.0

# Limiar de EAR para identificar sonolência
ear_limiar = 0.27
# Limiar de MAR para identificar boca aberta
mar_limiar = 0.5

# Inicialização da captura de vídeo
cap = cv2.VideoCapture(0)

# Utilizando o MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Variáveis para exibir a mensagem "Boca aberta" e "Boca fechada"
mensagem_boca = ""
mensagem_boca_fechada = "Boca fechada"
tempo_mensagem = 0  # Armazena o tempo que a mensagem foi exibida
tempo_limite_mensagem = 2  # Tempo máximo (em segundos) para a mensagem desaparecer

som_tocado = False
piscadas = 0
olhos_fechados = False
tempo_olhos_fechados = 0  # Tempo acumulado de olhos fechados
ultimo_tempo_piscada = time.time()  # Tempo da última piscada

# Inicializa os canais de som
channel_ear = pygame.mixer.Channel(0)
channel_mar = pygame.mixer.Channel(1)

# Cores suaves para os pontos
cor_rosto = (180, 180, 180)  # Cinza claro para o rosto
cor_olho = (255, 255, 0)     # Amarelo suave para os olhos
cor_boca = (255, 102, 102)   # Vermelho suave para a boca
cor_fundo = (50, 50, 50)     # Cor de fundo do painel (escuro)

# Iniciando a captura do vídeo
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Frame vazio da câmera ignorado.')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if resultado.multi_face_landmarks:
            for face_landmarks in resultado.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=cor_rosto, thickness=1, circle_radius=1),  # Rosto com cor suave
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=cor_rosto, thickness=1, circle_radius=1)  # Conexões do rosto com cor suave
                )

                face = face_landmarks.landmark

                # Desenhando os olhos com uma cor suave (amarelo)
                for id_coord, coord_xyz in enumerate(face_landmarks.landmark):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, frame.shape[1], frame.shape[0])
                        if coord_cv:
                            cv2.circle(frame, coord_cv, 2, cor_olho, -1)  # Cor amarela para os olhos

                # Desenhando a boca com uma cor suave (vermelho suave)
                for id_coord, coord_xyz in enumerate(face_landmarks.landmark):
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, frame.shape[1], frame.shape[0])
                        if coord_cv:
                            cv2.circle(frame, coord_cv, 2, cor_boca, -1)  # Cor vermelha suave para a boca

                # Cálculos EAR e MAR
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)

                # Detecção de piscadas (EAR < limiar)
                if ear < ear_limiar and not olhos_fechados:
                    piscadas += 1
                    olhos_fechados = True
                    ultimo_tempo_piscada = time.time()  # Atualiza o tempo da última piscada
                elif ear >= ear_limiar:
                    olhos_fechados = False

                # Calculando o tempo de olhos fechados
                if olhos_fechados:
                    tempo_olhos_fechados = time.time() - ultimo_tempo_piscada
                else:
                    tempo_olhos_fechados = 0

                # Exibir EAR, MAR, piscadas e tempo de olhos fechados com fundo sólido
                altura, largura = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (largura, 100), cor_fundo, -1)  # Fundo sólido para as informações

                # Exibindo as informações na tela
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Piscadas: {piscadas}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                # Adicionando fundo sólido para o tempo de olhos fechados
                cv2.rectangle(frame, (0, 120), (largura, 160), cor_fundo, -1)  # Fundo sólido para o tempo

                # Exibindo o tempo de olhos fechados
                cv2.putText(frame, f"Tempo: {tempo_olhos_fechados:.2f} s", (10, 150), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                # Atualizar a mensagem de "Boca fechada"
                if mar < mar_limiar:
                    mensagem_boca_fechada = "Boca fechada"
                else:
                    mensagem_boca_fechada = "Boca aberta"
                    if not som_tocado:  # Tocar som apenas uma vez quando a boca abrir
                        channel_mar.play(audio_file)  # Som para boca aberta
                        som_tocado = True

                # Condição para tocar o som ao fechar os olhos
                if ear < ear_limiar:
                    if not channel_ear.get_busy():  # Verifica se o som já está tocando
                        channel_ear.play(audio_file2)  # Som para olhos fechados
                else:
                    channel_ear.stop()  # Parar som quando os olhos estão abertos

                # Exibir mensagem de "Boca aberta" ou "Boca fechada" com fundo sólido
                cv2.rectangle(frame, (0, 160), (largura, 200), cor_fundo, -1)  # Fundo sólido para a mensagem de boca
                cv2.putText(frame, f"{mensagem_boca_fechada}", (10, 190), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)  # Mensagem de boca

        # Exibe a imagem da câmera com os pontos desenhados e o painel de informações
        cv2.imshow('Camera', frame)

        # Condição para sair da captura
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Fecha a captura
cap.release()
cv2.destroyAllWindows()
