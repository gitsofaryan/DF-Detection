import cv2
import os

def extract_frames_from_videos(video_dir, output_dir, label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        if not os.path.isfile(video_path):
            continue

        video_capture = cv2.VideoCapture(video_path)
        count = 0
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break

            frame_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{label}_frame{count}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1

        video_capture.release()
        print(f"Extracted {count} frames from {video_name}")

if __name__ == "__main__":

    celeb_real_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\Celeb-DF\Celeb-real"
    celeb_synthesis_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\Celeb-DF\Celeb-synthesis"
    youtube_real_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\Celeb-DF\YouTube-real"
    
    celeb_real_output_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\frames\Celeb-real"
    celeb_synthesis_output_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\frames\Celeb-synthesis"
    youtube_real_output_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\frames\YouTube-real"

    
    os.makedirs(celeb_real_output_dir, exist_ok=True)
    os.makedirs(celeb_synthesis_output_dir, exist_ok=True)
    os.makedirs(youtube_real_output_dir, exist_ok=True)


    extract_frames_from_videos(celeb_real_dir, celeb_real_output_dir, 'celeb_real')
    extract_frames_from_videos(celeb_synthesis_dir, celeb_synthesis_output_dir, 'celeb_synthesis')
    extract_frames_from_videos(youtube_real_dir, youtube_real_output_dir, 'youtube_real')
