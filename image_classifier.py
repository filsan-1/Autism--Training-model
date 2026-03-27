import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset path
DATASET_PATH = r"C:\Users\HP\Downloads\archive (1)\AutismDataset"

IMG_SIZE = 64

images = []
labels = []

def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()

            images.append(img)
            labels.append(label)

# Load both classes
load_images_from_folder(os.path.join(DATASET_PATH, "Autistic"), 1)
load_images_from_folder(os.path.join(DATASET_PATH, "Non_Autistic"), 0)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print("Loaded data:", X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
