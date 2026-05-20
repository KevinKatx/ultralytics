from ultralytics import YOLO
model = YOLO(r"C:\Clone Repos\Modified-Ultralytics-for-CBAM-Implementation-of-YOLOv8\tests\runs\detect\train12\weights\best.pt")
model.track(source=r"C:\Users\Test\Documents\ThesisVideos\New_VideoDataset\Separate Turbidities\Turbidity 2(Slight Cloudiness).mp4", show=True, save=True)