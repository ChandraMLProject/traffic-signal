
import streamlit as st
from PIL import Image
import numpy as np
from model import get_model, load_weights
from utils import preprocess_image, get_class_name

st.title("🚦 Traffic Sign Recognition")

uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = get_model()
    model = load_weights(model)
    
    img_array = preprocess_image(image)
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"### Predicted Sign: {get_class_name(predicted_class)}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
