import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load class names (match the order of folders used in training)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Load model
model = tf.keras.models.load_model('potato_disease.h5')

st.title("Potato Disease Detector 🌿🥔")
st.markdown("Upload a leaf image to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Predicted Class: **{predicted_class}**")
