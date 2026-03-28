import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
data_directory = 'C:/Users/HP/Downloads/archive (1)/AutismDataset/consolidated'
# Assuming that images are to be loaded here; add the proper loading mechanism

# Preprocess data: fill in with actual loading and preprocessing steps
# e.g., images_array, labels_array = load_and_preprocess_data(data_directory)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
# Test_loss, Test_accuracy = model.evaluate(X_test, y_test)
# print('Test accuracy:', Test_accuracy)