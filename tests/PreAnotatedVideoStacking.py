import cv2

# Load 4 videos
cap1 = cv2.VideoCapture(r'C:\Users\Test\Documents\ThesisVideos\LateBboneDemo.mp4')
cap2 = cv2.VideoCapture(r'C:\Users\Test\Documents\ThesisVideos\EarlyBboneDemo.mp4')
cap3 = cv2.VideoCapture(r'C:\Users\Test\Documents\ThesisVideos\YOLOv8Demo.mp4')
cap4 = cv2.VideoCapture(r'C:\Users\Test\Documents\ThesisVideos\YOLO11Demo.mp4')






# Define output resolution for each of the 4 videos
width, height = 640, 360 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\Test\Documents\ThesisVideos\stacked_output.mp4', fourcc, 24,(width * 2, height * 2))

while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened()):
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    ret3, f3 = cap3.read()
    ret4, f4 = cap4.read()

    if not (ret1 and ret2 and ret3 and ret4):
        break # Stop if any video ends

    # Resize all frames
    f1 = cv2.resize(f1,(width, height))
    f2 = cv2.resize(f2,(width, height))
    f3 = cv2.resize(f3,(width, height))
    f4 = cv2.resize(f4,(width, height))

  
    # Stack vertically (2 pairs)
    top_pair = cv2.vconcat([f1, f3])
    bottom_pair = cv2.vconcat([f2, f4])

    # Stack horizontally
    final_frame = cv2.hconcat([top_pair, bottom_pair])

    out.write(final_frame)
    cv2.imshow('2x2 Stack', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
out.release()
cv2.destroyAllWindows()
