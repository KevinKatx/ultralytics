from ultralytics import YOLO


model = YOLO('yolov8n.yaml')  # build a new model from YAML

results = model.train(data='coco128.yaml', epochs=3, imgsz=320)  # train the model