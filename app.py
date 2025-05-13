import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model("model/brain_tumor_classifier.h5")

# Define class names (update with your actual class names)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def predict(image):
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    predictions = model.predict(image_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {predicted_class: float(confidence)}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Brain Tumor Classifier",
    description="Upload a brain MRI image to classify the tumor type."
)

if __name__ == "__main__":
    interface.launch()
