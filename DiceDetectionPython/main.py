import cv2
import os

video_path = 'demo4.mp4'  # качи го в Colab
frames_to_extract = [90, 241, 453, 816]
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
for frame_num in frames_to_extract:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{output_dir}/frame_{frame_num}.jpg", frame)

cap.release()
