import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# TRAFFIC SIGN NAMES
# ============================================================

class_names = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Intelligent Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Intelligent Traffic Sign Recognition")

st.write(
    "CNN-based traffic sign classification using the "
    "GTSRB dataset."
)

st.divider()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    return load_model("traffic_sign_model.h5")


model = load_trained_model()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("Upload a Traffic Sign")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Read image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    # Convert image to NumPy array
    image_array = np.array(image)

    # Convert RGB → BGR
    # Same format used during training
    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # Resize
    image_array = cv2.resize(
        image_array,
        (32, 32)
    )

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Predicted class
    predicted_class = np.argmax(
        prediction
    )

    # Confidence
    confidence = (
        np.max(prediction) * 100
    )

    st.divider()

    # Result
    st.subheader("Prediction")

    st.success(
        f"🚦 {class_names[predicted_class]}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.caption(
    "Model: Simple CNN | "
    "Dataset: GTSRB | "
    "Classes: 43"
)