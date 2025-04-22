
import streamlit as st
from PIL import Image
import numpy as np
from model import get_model, load_weights
from utils import preprocess_image, get_class_name

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Traffic Sign Recognition")

st.markdown("Upload an image of a traffic sign, and the model will predict the sign category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Analyzing the image..."):
        model = get_model()
        model = load_weights(model)

        img_array = preprocess_image(image)
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.success(f"**Prediction:** {get_class_name(predicted_class)}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
