import cv2
from ultralytics import YOLO
import sys


#video = cv2.VideoCapture('video1.mp4')
modelo = YOLO("yolo11x_custom.pt")




source = input("Escolha a origem do teste: [1] Imagem [2] Video [3] Camera: ")

match source:
    case "1":
        print(f"{source} - validando uma imagem.")
        #model.predict(source='img.jpg', show=True, save=True)
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
    #print(img.shape) (1080, 1920, 3)
    #img = cv2.resize(img,(1270,720))1280
    img = cv2.resize(img,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)

    resultado = modelo(img)

    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            x,y,w,h = dados.xyxy[0]
            x,y,w,h = int(x),int(y),int(w),int(h)
            cv2.rectangle(img,(x,y),(w,h),(255,0,0),5)
            #cls = int(dados.cls[0])

            # if cls == 43:
            #     cv2.rectangle(img,(x,y),(w,h),(255,0,0),5)


    cv2.imshow('img', img)
    cv2.waitKey(1)
