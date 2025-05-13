import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np

# Load model
model = load_model("model/brain_tumor_classifier.h5")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Update if needed

# Set Streamlit page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classification")
st
