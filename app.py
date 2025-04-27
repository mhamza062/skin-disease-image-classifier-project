# skin_disease_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# --- Function to set custom background ---
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Function to preprocess image ---
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)
    if image.shape[-1] == 4:  # If the image has alpha channel, remove it
        image = image[..., :3]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_mobileNet_pso_eczema_specified_images.h5')
    return model

model = load_model()

# --- Set background ---
#set_background('background.jpg')  # Make sure you have a 'background.jpg' file

# --- App Title ---
st.markdown("<h1 style='text-align: center; color: white;'>🩺 Skin Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Upload a skin image to predict Eczema or Psoriasis</h4>", unsafe_allow_html=True)
st.write("")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])

# --- Main Prediction Section ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict Disease', use_container_width=True):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        class_names = ['Eczema', 'Psoriasis']
        
        if prediction > 0.5:
            result = class_names[1]  # Psoriasis
            confidence = prediction
        else:
            result = class_names[0]  # Eczema
            confidence = 1 - prediction
        
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

# --- Tips Section ---
st.markdown("---")
st.subheader("📸 Tips for Clear Skin Photos:")
st.markdown(
    """
- Take the photo in **bright natural light**.
- Ensure the **affected area** is **centered** and **in focus**.
- Avoid heavy makeup or creams during the photo.
- Capture a **clear, close-up** shot of the skin.
"""
)

# --- Disclaimer Section ---
st.markdown("---")
st.subheader("⚠️ Disclaimer:")
st.markdown(
    """
This app provides **AI-based predictions** only and is **not a substitute for professional medical advice**.  
For accurate diagnosis and treatment, always consult a **qualified dermatologist**.
"""
)
