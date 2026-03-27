import argparse
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

IMG_SIZE = 128
MODEL_OUTPUT_PATH = 'autism_cnn_model.keras'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN to classify autism vs non-autism images.')
    parser.add_argument(
        '--dataset-path',
        default=os.environ.get('DATASET_PATH', 'dataset'),
        help='Path to the dataset directory containing Autistic/ and Non_Autistic/ subfolders.',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum number of training epochs.')
    return parser.parse_args()


def load_images(dataset_path):
    """Load images from Autistic/ (label=1) and Non_Autistic/ (label=0) subfolders."""
    images = []
    labels = []

    class_map = {'Autistic': 1, 'Non_Autistic': 0}

    for class_name, label in class_map.items():
        folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(folder):
            logger.error('Folder does not exist: %s', folder)
            sys.exit(1)

        filenames = [f for f in os.listdir(folder) if not f.startswith('.')]
        if not filenames:
            logger.warning('No files found in folder: %s', folder)
            continue

        loaded = 0
        for filename in filenames:
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
                loaded += 1
            except Exception as exc:
                logger.warning('Skipping %s: %s', img_path, exc)

        logger.info('Loaded %d images from %s (label=%d)', loaded, folder, label)

    return np.array(images, dtype='float32'), np.array(labels, dtype='int32')


def build_model():
    """Build a CNN with 3 conv blocks and L2 regularization for binary classification."""
    l2 = regularizers.l2(1e-4)

    model = models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=l2),
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy'],
    )
    return model


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    epochs = args.epochs

    logger.info('Dataset path: %s', dataset_path)
    logger.info('Batch size: %d, Max epochs: %d', batch_size, epochs)

    # Load images
    images, labels = load_images(dataset_path)
    if len(images) == 0:
        logger.error('No images were loaded. Check the dataset path and folder structure.')
        sys.exit(1)

    logger.info('Total images: %d', len(images))

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info('Train samples: %d, Test samples: %d', len(X_train), len(X_test))

    # Compute class weights to handle imbalance
    unique_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights = dict(zip(unique_classes.tolist(), weights.tolist()))
    logger.info('Class weights: %s', class_weights)

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # Build model
    model = build_model()
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # Train
    logger.info('Starting training...')
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    logger.info('Evaluating on test set...')
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info('Test loss: %.4f', test_loss)
    logger.info('Test accuracy: %.4f (%.1f%%)', test_acc, test_acc * 100)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    logger.info('Confusion matrix:\n%s', cm)

    report = classification_report(y_test, y_pred, target_names=['Non_Autistic', 'Autistic'])
    logger.info('Classification report:\n%s', report)

    # Save model
    model.save(MODEL_OUTPUT_PATH)
    logger.info('Model saved to %s', MODEL_OUTPUT_PATH)


if __name__ == '__main__':
    main()
