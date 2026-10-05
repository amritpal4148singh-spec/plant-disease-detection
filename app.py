import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json


# ==============================
# Load Model
# ==============================

model = tf.keras.models.load_model(
    "model/plant_disease_38class.h5"
)


# ==============================
# Load Class Names
# ==============================

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)


# ==============================
# Image Configuration
# ==============================

IMG_SIZE = 192


# ==============================
# Streamlit UI
# ==============================

st.title("🌿 Plant Disease Detection App")

st.write(
    "Upload an image of a plant leaf to detect the possible disease."
)


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==============================
# Prediction
# ==============================

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # --------------------------
    # Preprocessing
    # --------------------------

    img = image.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(
        img,
        dtype=np.float32
    )

    arr = np.expand_dims(arr, axis=0)

    # EfficientNet preprocessing
    from tensorflow.keras.applications.efficientnet import preprocess_input

    arr = preprocess_input(arr)


    # --------------------------
    # Model Prediction
    # --------------------------

    preds = model.predict(arr, verbose=0)

    idx = np.argmax(preds[0])

    confidence = preds[0][idx]

    result = class_names[idx]


    # --------------------------
    # Display Result
    # --------------------------

    st.success(
        f"Prediction: {result}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # --------------------------
    # Top 3 Predictions
    # --------------------------

    st.subheader("Top Predictions")

    top3 = preds[0].argsort()[-3:][::-1]

    for i in top3:

        st.write(
            f"**{class_names[i]}** : "
            f"{preds[0][i] * 100:.2f}%"
        )