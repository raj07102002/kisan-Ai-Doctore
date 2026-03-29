import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Kisan AI Doctor",
    page_icon="🌾",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_crop_model.keras")

model = load_model()

class_names = ['Healthy', 'Powdery', 'Rust']

# ---------------- HEADER ----------------
st.title("🌾 Kisan AI Doctor")
st.markdown("### 📷 पत्ते की साफ फोटो अपलोड करें")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Leaf Image Upload Kare (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)


        # Preprocess
        img_resized = img.resize((244, 244))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        st.markdown("---")
        st.subheader("🔍 Prediction Result") 

        st.write("Raw Prediction:", prediction)
        st.write("Confidence Value:", confidence)

        # Confidence Bar
        st.progress(int(confidence))

        if confidence < 40:
            st.warning("⚠️ Model unsure hai. Clear image upload karein.")
        else:
            st.success(f"🌿 Disease: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")


            st.markdown("### 💊 Treatment Suggestion")

            if predicted_class == "Powdery":
                st.write("• Sulfur based fungicide spray karein.")
                st.write("• 7 din baad repeat karein.")
            elif predicted_class == "Rust":
                st.write("• Copper fungicide use karein.")
                st.write("• Affected leaves hata dein.")
            else:
                st.success("🌱 Plant healthy hai. Koi disease detect nahi hui.")

    except Exception as e:
        st.error("❌ Image process nahi ho payi. Dusri image try karein.")

st.markdown("---")
st.caption("Developed by Raj 🌾 | AI Powered Crop Disease Detection")
