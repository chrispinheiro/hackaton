import cv2
from ultralytics import YOLO
import winsound
import threading
import sys

alarmeCtl = False

source = input("Escolha a origem do teste: [1] Imagem [2] Video [3] Camera: ")
modelo = YOLO("yolo11x_custom.pt")

def alarme():
    for _ in range(5):
        winsound.Beep(2500,500)
    alarmeCtl = False

match source:
    case "1":
        print(f"{source} - validando uma imagem.")
        #model.predict(source='img.jpg', show=True, save=True)
        sys.exit()
    case "2":
        print(f"{source} - validando um vídeo.")
        video = cv2.VideoCapture('video1.mp4')
    case "3":
        print(f"{source} - validando câmera.")
        modelo.predict(source='0', show=True)
        video = cv2.VideoCapture(0)
    case _:
        print("That's not a valid value.")

#sys.exit()


while True:
    check, img = video.read()
    #print(img.shape) #(1080, 1920, 3)
    #img = cv2.resize(img,(1270,720))1280
    img = cv2.resize(img,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)

    resultado = modelo(img)

    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            conf = dados.conf.item()
            print(conf)
            if conf >= 0.5:
                x,y,w,h = dados.xyxy[0]
                x,y,w,h = int(x),int(y),int(w),int(h)
                cv2.rectangle(img,(x,y),(w,h),(255,0,0),5)
                cv2.rectangle(img,(100,30),(440,80),(0,0,255),-1)
                cv2.putText(img,'Detectado Objeto Cortante!',(105,65),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
                if not alarmeCtl:
                    alarmeCtl = True
                    threading.Thread(target=alarme).start()


    cv2.imshow('img', img)
    cv2.waitKey(1)
