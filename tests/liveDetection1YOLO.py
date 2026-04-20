import numpy as np
import cv2 as cv
from ultralytics import YOLO

model = YOLO(r"C:\Clone Repos\Modified-Ultralytics-for-CBAM-Implementation-of-YOLOv8\tests\runs\detect\Underwater Training Results\YOLOv11\weights\best.pt")

cap = cv.VideoCapture(r"C:\Users\Test\Documents\ThesisVideos\VideoDatasetFinal.mp4")
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(r"C:\Users\Test\Documents\ThesisVideos\Demo.mp4", fourcc, 24.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    font = cv.FONT_HERSHEY_SIMPLEX
    Text = cv.putText(frame,'YOLO11',(450,700), font, 2,(167, 121, 204),2,cv.LINE_AA)
    result = model(frame)

    out.write(result[0].plot())  # Write the frame with detections to the output video
    cv.imshow('frame', Text)
    cv.imshow('frame', result[0].plot())  # Display the frame with detections
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()