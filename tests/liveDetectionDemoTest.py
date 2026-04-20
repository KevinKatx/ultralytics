from ultralytics import YOLO
model = YOLO(r"C:\Clone Repos\Modified-Ultralytics-for-CBAM-Implementation-of-YOLOv8\tests\runs\detect\Underwater Training Results\YOLOv8earlyBBone\weights\best.pt")
model.track(source=r"C:\Users\Test\Documents\ThesisVideos\VideoDataset.mp4", show=True, save=True)