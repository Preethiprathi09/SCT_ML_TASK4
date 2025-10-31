# app.py for Hand Gesture Prediction (Task4)

import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load your trained SVM model
with open("gesture_model.pkl", "rb") as file:
    model = joblib.load(file)

st.title("Hand Gesture Prediction")

# Upload image
uploaded_file = st.file_uploader("Upload a hand gesture image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # 1️⃣ Convert to grayscale (if your model was trained on grayscale)
    img = img.convert('L')
    
    # 2️⃣ Resize to 28x28 (match training data)
    img = img.resize((28, 28))
    
    # 3️⃣ Flatten and reshape to (1, 784)
    X_input = np.array(img).flatten().reshape(1, -1)
    
    # 4️⃣ Optional: normalize if training images were normalized
    X_input = X_input / 255.0
    
    # 5️⃣ Predict
    prediction = model.predict(X_input)[0]
    
    st.success(f"Predicted Gesture: {prediction}")
