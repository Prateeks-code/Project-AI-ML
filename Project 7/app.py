import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Gender Recognition",
    page_icon="👨",
    layout="centered"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

MODEL_PATH = "gender_model.keras"   # Rename your model file to this

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------------
# TITLE
# -----------------------------------

st.markdown(
    "<h1 style='text-align:center;color:#1E90FF;'>Gender Recognition using CNN</h1>",
    unsafe_allow_html=True,
)

st.write(
    "Upload an image and the model will predict whether the person is **Male** or **Female**."
)

st.divider()

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"],
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize according to model input
    image = image.resize((100, 100))

    img_array = np.array(image).astype("float32")

    img_array /= 255.0

    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):

        prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    st.divider()

    if probability < 0.5:

        confidence = (1 - probability) * 100

        st.success("Prediction: Male")
        st.info(f"Confidence: {confidence:.2f}%")

    else:

        confidence = probability * 100

        st.success("Prediction: Female")
        st.info(f"Confidence: {confidence:.2f}%")

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.markdown(
    "<center>Made with ❤️ using Streamlit and TensorFlow</center>",
    unsafe_allow_html=True,
)
