import streamlit as st
import os
from PIL import Image
import random

# Dataset directory (update this path as needed)
DATASET_DIR = "Dataset_final/test"

# Mapping numeric folder names to emotion labels
emotion_labels = {
    "1": "Surprise",
    "2": "Disgust",
    "3": "Happiness",
    "4": "Sadness",
    "5": "Anger",
    "6": "Neutral"
}

def show_sample_images_page():
    st.title("Face Emotion Dataset Visualization")

    # Slider to control number of images per emotion
    num_images = st.slider("Number of images to display per emotion:", min_value=1, max_value=20, value=5)

    # Check dataset path
    if not os.path.isdir(DATASET_DIR):
        st.error(f"Dataset path '{DATASET_DIR}' not found.")
    else:
        # Only process folders that are in the defined emotion_labels
        valid_folders = [f for f in os.listdir(DATASET_DIR) if f in emotion_labels]

        if not valid_folders:
            st.warning("No valid emotion folders (1–6) found in the dataset directory.")
        else:
            for folder in sorted(valid_folders):
                emotion_name = emotion_labels[folder]
                emotion_path = os.path.join(DATASET_DIR, folder)
                image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                st.subheader(f"{emotion_name} ({len(image_files)} images)")
                selected_images = random.sample(image_files, min(num_images, len(image_files)))

                cols = st.columns(min(5, len(selected_images)))
                for i, img_file in enumerate(selected_images):
                    img_path = os.path.join(emotion_path, img_file)
                    try:
                        image = Image.open(img_path)
                        cols[i % len(cols)].image(image, caption=img_file)
                    except Exception as e:
                        st.error(f"Failed to load {img_path}: {e}")