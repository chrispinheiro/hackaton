Estrutura do Código
O pipeline é dividido em quatro etapas principais:

Detecção e Criação do Dataset.
Treinamento do Modelo YOLO com o Dataset.
Sistema de Inferência com Alerta por E-mail.
Configurações de Dependências e Ambiente.
Passo 1: Instale as Dependências
Instale as bibliotecas necessárias:

pip install ultralytics opencv-python smtplib

CODIGO COMPLETO

import cv2
import os
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
import shutil
from pathlib import Path


# 1. CONFIGURAR O ALERTA POR E-MAIL
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "seu_email@gmail.com"  # Substitua pelo seu e-mail
EMAIL_PASSWORD = "sua_senha"       # Substitua pela sua senha
ALERT_RECIPIENT = "destinatario@gmail.com"  # Substitua pelo destinatário

def send_email_alert(video_name, frame_number):
    """Envia um alerta por e-mail."""
    try:
        msg = MIMEText(f"Objeto cortante detectado no vídeo '{video_name}' no frame {frame_number}!")
        msg["Subject"] = "Alerta: Objeto Cortante Detectado"
        msg["From"] = EMAIL_USER
        msg["To"] = ALERT_RECIPIENT

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, ALERT_RECIPIENT, msg.as_string())
        print("Alerta enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar o alerta: {e}")


# 2. DETECÇÃO E GERAÇÃO DE DATASET
def process_video(video_path, output_folder, model_path="yolov8n.pt"):
    """
    Processa o vídeo para identificar objetos cortantes e gerar um dataset.
    Cria imagens positivas (com objetos) e negativas (sem objetos).
    """
    # Carregar o modelo YOLO pré-treinado
    model = YOLO(model_path)

    # Criar pastas para o dataset
    positive_dir = Path(output_folder) / "positives"
    negative_dir = Path(output_folder) / "negatives"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    # Processar o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Fazer a detecção com YOLO
        results = model(frame)
        boxes = results[0].boxes  # Coordenadas dos objetos detectados

        if len(boxes) > 0:
            # Salvar frames com objetos detectados (positivos)
            cv2.imwrite(str(positive_dir / f"frame_{frame_count}.jpg"), frame)

            # Enviar alerta por e-mail
            send_email_alert(video_path, frame_count)
        else:
            # Salvar frames sem objetos detectados (negativos)
            cv2.imwrite(str(negative_dir / f"frame_{frame_count}.jpg"), frame)

    cap.release()
    print("Processamento do vídeo concluído!")
    print(f"Dataset gerado em: {output_folder}")


# 3. TREINAR O MODELO YOLO COM O DATASET
def train_yolo(dataset_path, data_yaml, model_name="yolov8n.pt", epochs=50, batch_size=16):
    """
    Treina um modelo YOLO com o dataset gerado.
    """
    model = YOLO(model_name)
    model.train(
        data=data_yaml,  # Arquivo data.yaml com o caminho do dataset
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name="sharp_objects_detection"
    )
    print("Treinamento concluído! Modelo salvo na pasta 'runs/train'.")


# 4. INFERÊNCIA E ALERTA EM TEMPO REAL
def inference_with_alert(video_path, model_path="runs/train/sharp_objects_detection/weights/best.pt"):
    """
    Executa inferências em um novo vídeo e envia alertas por e-mail se objetos cortantes forem detectados.
    """
    model = YOLO(model_path)

    # Processar o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Fazer a inferência
        results = model(frame)
        boxes = results[0].boxes

        if len(boxes) > 0:
            print(f"Objeto cortante detectado no frame {frame_count}. Enviando alerta...")
            send_email_alert(video_path, frame_count)

    cap.release()
    print("Inferência concluída!")


# 5. CONFIGURAÇÕES E EXECUÇÃO
if __name__ == "__main__":
    # Caminhos
    VIDEO_PATH = "video.mp4"  # Substitua pelo caminho do seu vídeo
    OUTPUT_FOLDER = "dataset"
    DATA_YAML = "data.yaml"

    # Passo 1: Gerar o dataset
    process_video(VIDEO_PATH, OUTPUT_FOLDER)

    # Passo 2: Criar o arquivo data.yaml
    with open(DATA_YAML, "w") as f:
        f.write(f"""
train: {OUTPUT_FOLDER}/positives
val: {OUTPUT_FOLDER}/negatives
nc: 1
names: ['sharp_objects']
        """)

    # Passo 3: Treinar o modelo YOLO
    train_yolo(OUTPUT_FOLDER, DATA_YAML)

    # Passo 4: Inferir e enviar alertas em um novo vídeo
    inference_with_alert(VIDEO_PATH)

--------------

Explicação do Código
Detecção e Dataset:

O vídeo é dividido em frames.
Os frames com objetos detectados são salvos como positivos.
Os frames sem detecções são salvos como negativos.
Treinamento:

O arquivo data.yaml configura o dataset no formato YOLO.
O modelo YOLO é treinado com os frames positivos e negativos para melhorar a precisão.
Inferência e Alerta:

O modelo treinado é usado para processar um novo vídeo.
Caso objetos cortantes sejam detectados, um e-mail é enviado automaticamente.
Passo 3: Executar o Pipeline
Substitua os caminhos do vídeo, pastas e e-mails no código.
Execute o script.
python sharp_objects_pipeline.py