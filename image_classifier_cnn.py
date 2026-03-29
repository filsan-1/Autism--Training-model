import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ImageClassifierCNN:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, validation_data, class_weight=None):
        early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=3)
        model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

        history = self.model.fit(train_data,
                                  validation_data=validation_data,
                                  epochs=50,
                                  class_weight=class_weight,
                                  callbacks=[early_stopping, reduce_lr, model_checkpoint])
        return history

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

# Add data augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
 
# Update other parts of the training script accordingly.

# Evaluation utility module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='best')
    plt.show()


# Similarly, functions for precision-recall curves and other metrics can be added.
