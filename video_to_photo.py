import cv2
import os
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
import shutil
from pathlib import Path


photo_dir = "./photos"

# Processar o vídeo
cap = cv2.VideoCapture("video2.mp4")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    cv2.imwrite(f"photos/frame_{frame_count}.jpg", frame)

cap.release()
print("Processamento do vídeo concluído!")