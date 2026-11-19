import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dataset/Train"
IMG_SIZE = 32
NUM_CLASSES = 43


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = load_model("traffic_sign_model.h5")

print("Model loaded successfully!")


# ============================================================
# LOAD DATASET
# ============================================================

images = []
labels = []

print("\nLoading images...")


for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    label = int(class_name)

    for image_name in os.listdir(class_path):

        image_path = os.path.join(
            class_path,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        images.append(image)
        labels.append(label)


images = np.array(images)
labels = np.array(labels)

print("Total images:", len(images))


# ============================================================
# NORMALIZE
# ============================================================

images = images / 255.0


# ============================================================
# CREATE TEST SET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    images,
    labels,

    test_size=0.2,

    random_state=42,

    stratify=labels
)


print("Test images:", len(X_test))


# ============================================================
# PREDICTIONS
# ============================================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

# Get class with highest probability
y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Traffic Sign Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.colorbar()

plt.xticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES),
    rotation=90
)

plt.yticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES)
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("        CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


print("\n========================================")
print("          TESTING COMPLETED")
print("========================================")

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")