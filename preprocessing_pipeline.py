import os
import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_and_crop_face(image_array):
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if faces:
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = image_rgb[y:y+height, x:x+width]
        return cropped_face
    else:
        return image_rgb

def preprocess_image(image_array):
    try:
        cropped_face = detect_and_crop_face(image_array)
        resized_face = cv2.resize(cropped_face, (128, 128))
        normalized_face = resized_face / 255.0
        return normalized_face
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def preprocess_directory(directory_path, save_to_dir=None, max_images=None):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if save_to_dir and not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    image_count = 0
    for filename in os.listdir(directory_path):
        if max_images is not None and image_count >= max_images:
            break
        
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            if image is not None:
                processed_image = preprocess_image(image)
                if processed_image is not None and save_to_dir:
                    save_path = os.path.join(save_to_dir, filename)
                    cv2.imwrite(save_path, cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                image_count += 1

if __name__ == "__main__":
    base_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data"
    
    celeb_real_frames_dir = os.path.join(base_dir, "frames", "Celeb-real")
    celeb_synthesis_frames_dir = os.path.join(base_dir, "frames", "Celeb-synthesis")
    youtube_real_frames_dir = os.path.join(base_dir, "frames", "YouTube-real")
    
    processed_output_dir = os.path.join(base_dir, "processed_frames")

    max_images_to_process = 100 
    preprocess_directory(celeb_real_frames_dir, save_to_dir=os.path.join(processed_output_dir, 'Celeb-real'), max_images=max_images_to_process)
    preprocess_directory(celeb_synthesis_frames_dir, save_to_dir=os.path.join(processed_output_dir, 'Celeb-synthesis'), max_images=max_images_to_process)
    preprocess_directory(youtube_real_frames_dir, save_to_dir=os.path.join(processed_output_dir, 'YouTube-real'), max_images=max_images_to_process)
