import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Apply focal loss formula
        return tf.reduce_mean(alpha * tf.pow(1 - y_pred, gamma) * cross_entropy)
    return focal_loss_fixed


def weighted_class_loss(weights):
    def loss(y_true, y_pred):
        # Calculate class weights loss
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return -tf.reduce_mean(weights * y_true * tf.math.log(y_pred))
    return loss


def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model


# Data Augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Compile model with optimized loss
model = create_model()
model.compile(optimizer='adam', 
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy']) 

# Training the model
model.fit(train_data, train_labels, epochs=50, class_weight={0:1, 1:2}, callbacks=[...])  # Add necessary callbacks
