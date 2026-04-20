import cv2
import os

def extract_frames(video_path, output_folder):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    while True:
        # 3. Read frame-by-frame
        ret, frame = cap.read()

        # If ret is False, the video has ended
        if not ret:
            break

        # 4. Save the current frame
        if frame_count%3==0:
            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    # 5. Release resources
    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to '{output_folder}'.")

# Example usage
extract_frames(r'C:\Users\Test\Documents\ThesisVideos\VideoDatasetSegmented\NTU160_190.mp4', r'C:\Users\Test\Documents\ThesisVideos\VideoDatasetSegmented\NTU160_190')
