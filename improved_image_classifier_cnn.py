import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score

# Custom F1 Score Callback
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = (self.model.predict(x_val) > 0.5).astype("int32")
        f1 = f1_score(y_val, y_pred)
        print(f"F1 Score for epoch {epoch}: {f1}")

# Build Model
def build_model(input_shape):
    model = models.Sequential()
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Preparing model
input_shape = (128, 128, 3)  # Example input shape
model = build_model(input_shape)
class_weights = {0: 1.0, 1: 2.0}
model.compile(optimizer='adam', 
              loss=tf.keras.losses.binary_crossentropy, 
              metrics=['accuracy'])

# Assuming x_train, y_train, x_val, y_val are defined
f1_callback = F1ScoreCallback(validation_data=(x_val, y_val))

# Train Model
model.fit(x_train, y_train, 
          validation_data=(x_val, y_val), 
          epochs=50, 
          batch_size=32, 
          class_weight=class_weights, 
          callbacks=[f1_callback])