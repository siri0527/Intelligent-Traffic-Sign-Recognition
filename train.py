import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "dataset/Train"

IMG_SIZE = 32
NUM_CLASSES = 43


# ============================================================
# 2. LOAD DATASET
# ============================================================

images = []
labels = []

print("========================================")
print("      INTELLIGENT TRAFFIC SIGN")
print("          RECOGNITION SYSTEM")
print("========================================")

print("\nLoading images...")


for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    # Ignore files
    if not os.path.isdir(class_path):
        continue

    # Convert folder name to class number
    label = int(class_name)

    print("Loading class:", label)

    # Read images
    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        # Read image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        images.append(image)
        labels.append(label)


# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)


print("\nDataset loaded successfully!")

print("Total images:", len(images))
print("Image shape:", images.shape)
print("Label shape:", labels.shape)


# ============================================================
# 3. NORMALIZE IMAGES
# ============================================================

print("\nNormalizing images...")

# Convert pixel values from 0-255 to 0-1
images = images / 255.0


# ============================================================
# 4. SPLIT DATASET
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ============================================================
# 5. CREATE CNN MODEL
# ============================================================

print("\nCreating CNN model...")


model = Sequential([

    # First convolution layer
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Pooling
    MaxPooling2D(
        (2, 2)
    ),


    # Second convolution layer
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Pooling
    MaxPooling2D(
        (2, 2)
    ),


    # Convert feature maps into a vector
    Flatten(),


    # Fully connected layer
    Dense(
        128,
        activation="relu"
    ),


    # Dropout to reduce overfitting
    Dropout(0.5),


    # Output layer
    # 43 traffic sign classes
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 6. DISPLAY MODEL
# ============================================================

print("\nCNN Model Summary:")

model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n========================================")
print("          STARTING TRAINING")
print("========================================\n")


history = model.fit(

    X_train,

    y_train,

    epochs=10,

    batch_size=32,

    validation_data=(
        X_test,
        y_test
    )
)


# ============================================================
# 9. PLOT ACCURACY
# ============================================================

print("\nCreating accuracy graph...")


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    "accuracy.png"
)

plt.show()


# ============================================================
# 10. PLOT LOSS
# ============================================================

print("\nCreating loss graph...")


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    "loss.png"
)

plt.show()


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

print("\n========================================")
print("          EVALUATING MODEL")
print("========================================")


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)


print("\nTest Accuracy:",
      test_accuracy)

print("Test Loss:",
      test_loss)


# ============================================================
# 12. SAVE MODEL
# ============================================================

print("\nSaving model...")


model.save(
    "traffic_sign_model.h5"
)


print("\n========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("========================================")

print("\nModel file:")
print("traffic_sign_model.h5")

print("\nGraphs created:")
print("accuracy.png")
print("loss.png")

print("\nFinal Test Accuracy:",
      f"{test_accuracy * 100:.2f}%")

print("\n========================================")
print("             PROJECT DONE")
print("========================================")