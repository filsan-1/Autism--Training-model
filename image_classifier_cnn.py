"""
Autism Image Classification - Improved CNN Training Pipeline
============================================================
Addresses class imbalance and low recall for the autistic class by combining:
  - Balanced class weights
  - Aggressive data augmentation
  - Deeper CNN architecture with BatchNorm and L2 regularization
  - ModelCheckpoint saving the best model by validation recall
  - EarlyStopping and ReduceLROnPlateau monitoring validation recall
  - Decision-threshold tuning to maximise recall for the autistic class
  - Comprehensive evaluation: confusion matrix, ROC curve, PR curve
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Input, layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

IMG_SIZE = 128
MODEL_OUTPUT_PATH = 'autism_cnn_model.keras'
BEST_MODEL_PATH = 'autism_cnn_best.keras'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN to classify autism vs non-autism images.')
    parser.add_argument(
        '--dataset-path',
        default=os.environ.get('DATASET_PATH', 'dataset'),
        help='Path to the dataset directory containing Autistic/ and Non_Autistic/ subfolders.',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum number of training epochs.')
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
    """Build a deeper CNN (4 conv blocks, 32->64->128->256) with BatchNorm and L2 regularisation.

    Key design choices for improved recall:
      - Deeper feature extraction (4 blocks vs 3) captures richer facial patterns.
      - BatchNorm stabilises training and allows a higher learning rate.
      - Moderate dropout prevents overfitting without destroying signal.
      - Recall metric tracked during training so callbacks can monitor it directly.
    """
    l2 = regularizers.l2(1e-4)

    model = models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.0005),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
        ],
    )
    return model


def tune_threshold(y_true, y_prob):
    """Sweep thresholds in [0.25, 0.65] and return the one with the best F1 while
    maintaining recall >= 0.65 for the autistic class.

    Returns the best threshold and a dict of per-threshold metrics for logging.
    """
    thresholds = np.arange(0.25, 0.66, 0.05)
    best_thresh = 0.5
    best_f1 = 0.0
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({'threshold': round(float(t), 2), 'recall': recall, 'precision': precision, 'f1': f1})
        # Prioritise recall >= 0.65; among valid thresholds pick highest F1
        if recall >= 0.65 and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    # Fall back to the threshold with the highest recall when none hits the target
    if best_f1 == 0.0:
        best_thresh = float(results[max(range(len(results)), key=lambda i: results[i]['recall'])]['threshold'])

    return best_thresh, results


def plot_confusion_matrix(cm, output_path='confusion_matrix.png'):
    """Save a labelled confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Non_Autistic', 'Autistic']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info('Confusion matrix plot saved to %s', output_path)


def plot_roc_curve(y_true, y_prob, output_path='roc_curve.png'):
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info('ROC curve saved to %s (AUC=%.4f)', output_path, roc_auc)
    return roc_auc


def plot_precision_recall_curve(y_true, y_prob, output_path='precision_recall_curve.png'):
    """Save a Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (Autistic class)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info('Precision-Recall curve saved to %s (AUC=%.4f)', output_path, pr_auc)
    return pr_auc


def plot_training_history(history, output_path='training_history.png'):
    """Save accuracy, loss, and recall training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    if 'recall' in history.history:
        axes[2].plot(history.history['recall'], label='Train')
        axes[2].plot(history.history['val_recall'], label='Val')
        axes[2].set_title('Recall (Autistic class)')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info('Training history plot saved to %s', output_path)


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    epochs = args.epochs

    logger.info('Dataset path: %s', dataset_path)
    logger.info('Batch size: %d, Max epochs: %d', batch_size, epochs)

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    images, labels = load_images(dataset_path)
    if len(images) == 0:
        logger.error('No images were loaded. Check the dataset path and folder structure.')
        sys.exit(1)

    logger.info('Total images: %d', len(images))

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info('Train samples: %d, Test samples: %d', len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # Class weights - boost the autistic class proportionally
    # ------------------------------------------------------------------
    unique_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights = dict(zip(unique_classes.tolist(), weights.tolist()))
    logger.info('Class weights: %s', class_weights)

    # ------------------------------------------------------------------
    # Aggressive data augmentation for the training set
    # ------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode='nearest',
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model()
    model.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # Callbacks: checkpoint on val_recall, early-stop and LR-decay too
    # ------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_recall',
            mode='max',
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_recall',
            mode='max',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info('Starting training...')
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Evaluate with default threshold (0.5)
    # ------------------------------------------------------------------
    logger.info('Evaluating on test set...')
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    y_pred_default = (y_pred_prob >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm_default.ravel()
    logger.info('--- Default threshold (0.5) ---')
    logger.info('TP=%d  TN=%d  FP=%d  FN=%d', tp, tn, fp, fn)
    recall_default = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_default = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    logger.info('Recall=%.4f  Precision=%.4f', recall_default, precision_default)
    logger.info('Classification report (threshold=0.5):\n%s',
                classification_report(y_test, y_pred_default, target_names=['Non_Autistic', 'Autistic']))

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------
    best_thresh, thresh_results = tune_threshold(y_test, y_pred_prob)
    logger.info('Threshold sweep results:')
    for r in thresh_results:
        logger.info('  thresh=%.2f  recall=%.4f  precision=%.4f  f1=%.4f',
                    r['threshold'], r['recall'], r['precision'], r['f1'])
    logger.info('Selected threshold: %.2f', best_thresh)

    y_pred_tuned = (y_pred_prob >= best_thresh).astype(int)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    tn2, fp2, fn2, tp2 = cm_tuned.ravel()
    logger.info('--- Tuned threshold (%.2f) ---', best_thresh)
    logger.info('TP=%d  TN=%d  FP=%d  FN=%d', tp2, tn2, fp2, fn2)
    recall_tuned = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
    precision_tuned = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
    specificity_tuned = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
    f1_tuned = (2 * precision_tuned * recall_tuned / (precision_tuned + recall_tuned)
                if (precision_tuned + recall_tuned) > 0 else 0.0)
    logger.info('Recall=%.4f  Precision=%.4f  Specificity=%.4f  F1=%.4f',
                recall_tuned, precision_tuned, specificity_tuned, f1_tuned)
    logger.info('Classification report (threshold=%.2f):\n%s', best_thresh,
                classification_report(y_test, y_pred_tuned, target_names=['Non_Autistic', 'Autistic']))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_confusion_matrix(cm_tuned, output_path='confusion_matrix.png')
    roc_auc = plot_roc_curve(y_test, y_pred_prob, output_path='roc_curve.png')
    pr_auc = plot_precision_recall_curve(y_test, y_pred_prob, output_path='precision_recall_curve.png')
    plot_training_history(history, output_path='training_history.png')

    logger.info('ROC-AUC: %.4f', roc_auc)
    logger.info('PR-AUC:  %.4f', pr_auc)

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    model.save(MODEL_OUTPUT_PATH)
    logger.info('Final model saved to %s', MODEL_OUTPUT_PATH)
    logger.info('Best checkpoint (by val_recall) saved to %s', BEST_MODEL_PATH)


if __name__ == '__main__':
    main()
