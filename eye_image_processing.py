# eye_image_processing.py

"""
A simple Python pipeline for eye image processing.
This script loads images from a folder, resizes them,
detects faces and eyes using MediaPipe, and crops the eye region.
"""

import cv2
import os
import mediapipe as mp

# Initialize MediaPipe Face and Eye detectors
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Function to load images from a specified directory

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to resize images to a preferred width while maintaining aspect ratio

def resize_image(image, width=400):
    height = int((width / image.shape[1]) * image.shape[0])
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Function to detect and crop eye regions from images

def crop_eyes(images):
    eye_images = []
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        for image in images:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    # Cropping the face region
                    face_crop = image[y:y+h, x:x+w]  
                    # Assuming eye region is located in the lower part of the face crop
                    eye_crop = face_crop[int(h/2):h, 0:w]
                    eye_images.append(eye_crop)
    return eye_images

# Main execution
if __name__ == '__main__':
    image_folder = 'path/to/your/images'  # Change this to your image folder path
    images = load_images_from_folder(image_folder)
    resized_images = [resize_image(img) for img in images]
    eye_images = crop_eyes(resized_images)

    # Optionally, save eye images to a folder
    output_folder = 'eye_images/'
    os.makedirs(output_folder, exist_ok=True)
    for i, eye_img in enumerate(eye_images):
        cv2.imwrite(os.path.join(output_folder, f'eye_image_{i}.png'), eye_img)

    print(f'Processed {len(eye_images)} eye images. Check the output folder.')