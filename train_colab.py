# Conectar pasta do Google Drive
#/content/drive/MyDrive/hackaton/treino01
from google.colab import drive
drive.mount('/content/drive')

#-----

# Baixar do Roboflow as imagens baixadas e/ou anotadas
# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key=API_KEY_ROBOFLOW)
# project = rf.workspace("hackaton-8tea5").project("weapon-detection-knifes-mohip-mhhf9")
# version = project.version(1)
# dataset = version.download("yolov11")

#-----

!pip install ultralytics

#-----

from ultralytics import YOLO

model = YOLO('yolo11n_custom.pt')

model.train(data='dataset_custom.yaml',epochs=100,batch=16,imgsz=640)

#-----

