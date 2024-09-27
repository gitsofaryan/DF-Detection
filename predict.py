import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from preprocessing_pipeline import preprocess_image
import random


model = load_model(r'C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake_detection_model_lstm.h5')

def predict_deepfake(sequence_of_images):
    """Predict whether the given sequence of images contains a deepfake."""

    preprocessed_images = [preprocess_image(image) for image in sequence_of_images]
    

    preprocessed_sequence = np.stack(preprocessed_images, axis=0)
    

    preprocessed_sequence = np.expand_dims(preprocessed_sequence, axis=0)
    
    try:

        prediction = model.predict(preprocessed_sequence)
        return prediction[0][0]  
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":

    real_images_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\processed_frames\celeb-real"
    synthetic_images_dir = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\processed_frames\celeb-synthesis"
    
    
    num_images_to_sample = 5  
    

    def sample_images_from_directory(directory, num_samples):
        all_images = [os.path.join(directory, image_name) for image_name in sorted(os.listdir(directory)) if os.path.isfile(os.path.join(directory, image_name))]
        if len(all_images) >= num_samples:
            sampled_images_paths = random.sample(all_images, num_samples)
            images = []
            for image_path in sampled_images_paths:
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                else:
                    print(f"Failed to load image: {image_path}")
            return images
        else:
            print(f"Not enough images in {directory} to sample {num_samples} images.")
            return []


    real_images = sample_images_from_directory(real_images_dir, num_images_to_sample)
    synthetic_images = sample_images_from_directory(synthetic_images_dir, num_images_to_sample)

    combined_images = real_images + synthetic_images
    

    if combined_images:
        deepfake_prob = predict_deepfake(combined_images)
        
        if deepfake_prob is not None:
            print(f"Combined Images | Deepfake Probability: {deepfake_prob:.4f}")
        else:
            print(f"Failed to process combined images.")
    else:
        print("No valid images to process.")
