from ultralytics import YOLO
model = YOLO(r'C:\Clone Repos\Modified-Ultralytics-for-CBAM-Implementation-of-YOLOv8\ultralytics\cfg\models\v8\yolov8-CBAM.yaml')  # build a new model from YAML
results = model.train(data='Underwater.yaml', epochs=3, imgsz=320)  # train the model





###Terminal Command to run the test file
