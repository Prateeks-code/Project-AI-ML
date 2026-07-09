import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("male_female_model.pkl")

IMG_SIZE = 64

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Male/Female Classifier",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Male/Female Image Classifier")
st.write("Upload an image and the model will predict whether it is Male or Female.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL image to OpenCV format
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Flatten
    img = img.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(img)[0]

    # Probability (if available)
    try:
        probability = model.predict_proba(img)[0]
        confidence = np.max(probability) * 100
    except:
        confidence = None

    label = "Male" if prediction == 0 else "Female"

    st.success(f"Prediction: **{label}**")

    if confidence is not None:
        st.info(f"Confidence: **{confidence:.2f}%**")
