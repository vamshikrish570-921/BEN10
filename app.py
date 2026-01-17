import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ben 10 Character Predictor",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join("model", "ben10_model.h5")
model = load_model(MODEL_PATH)

# ---------------- LOAD LABELS ----------------
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🧬 Ben 10 Cartoon Character Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image and get the closest Ben 10 character</p>", unsafe_allow_html=True)
st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ---------------- IMAGE UPLOAD ----------------
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# ---------------- PREDICTION ----------------
if uploaded_file:
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index]) * 100
    character = labels[index]

    with col2:
        st.subheader("🧠 Prediction Result")

        # ---------- CARTOON IMAGE ----------
        cartoon_image_path = os.path.join(
            "character_images", f"{character}.png"
        )

        if os.path.exists(cartoon_image_path):
            cartoon_img = Image.open(cartoon_image_path)
            st.image(cartoon_img, caption=character, width=300)
        else:
            st.error("Character image not found!")

        # ---------- TEXT RESULT ----------
        st.markdown(f"### 🦸 Character: **{character}**")
        st.progress(int(confidence))
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        if confidence < 50:
            st.warning("Low confidence prediction. Try a different image.")
        else:
            st.success("Prediction successful!")

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)
