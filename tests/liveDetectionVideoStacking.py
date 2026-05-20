import cv2
from ultralytics import YOLO


def run_comparison_demo(file_path, model1_path, model2_path, model3_path, model4_path):
# Load 4 videos
    cap = cv2.VideoCapture(file_path)

    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)
    model3 = YOLO(model3_path)
    model4 = YOLO(model4_path)

    # Define output resolution for each of the 4 videos
    width, height = 640, 360 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'C:\Users\Test\Documents\ThesisVideos\stacked_output.mp4', fourcc, 24,(width * 2, height * 2))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break # Stop if any video ends

        # Resize all frames
        f1 = cv2.resize(frame.copy(),(width, height))
        f2 = cv2.resize(frame.copy(),(width, height))
        f3 = cv2.resize(frame.copy(),(width, height))
        f4 = cv2.resize(frame.copy(),(width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        result1 = model1(f1)
        result2 = model2(f2)
        result3 = model3(f3)
        result4 = model4(f4)

        img1 = result1[0].plot()
        img2 = result2[0].plot()
        img3 = result3[0].plot()
        img4 = result4[0].plot()

        cv2.putText(img1,'EarlyBbone',(10,50), font, 1,(178, 114, 0),2,cv2.LINE_AA)
        cv2.putText(img2,'LateBbone',(10,50), font, 1,(0, 159, 230),2,cv2.LINE_AA)
        cv2.putText(img3,'YOLOv8',(10,50), font, 1,(115, 158, 0),2,cv2.LINE_AA)
        cv2.putText(img4,'YOLO11',(10,50), font, 1,(167, 121, 204),2,cv2.LINE_AA)

        # Stack vertically (2 pairs)
        top_pair = cv2.vconcat([img1, img3])
        bottom_pair = cv2.vconcat([img2, img4])

        # Stack horizontally
        final_frame = cv2.hconcat([top_pair, bottom_pair])

        out.write(final_frame)
        cv2.imshow('2x2 Stack', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
