from ultralytics import YOLO

model = YOLO('yolo11l_custom.pt')

#model.predict(source='img.jpg', show=True, save=True)

model.predict(source='video1.mp4', show=True, save=True)
