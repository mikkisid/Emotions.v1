import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np

from pytorch_grad_cam.utils.image import show_cam_on_image

import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2



#import modules from this repository 
import models 
import prediction
import data_viz
import introduction
import dataset_viz





# -------------------------------
# Label Dictionary (1-indexed)
# -------------------------------
label_dict = {
    1: 'Surprise',
    2: 'Disgust',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Anger',
    6: 'Neutral'
}



# -------------------------------
# Streamlit App UI
# -------------------------------


st.set_page_config(page_title="Emotion Classification With Computer Vision", layout="centered")

st.title("🎭 Facial Expression Recognition")

# Model selection
model_choice = st.selectbox("Choose a model", ["CNN", "VGG16", "ViT"])

model = models.load_cnn_model()

app_mode = st.sidebar.selectbox('Contents ',['01 Introduction','02 Dataset visualization', '03 Metrics and Model Architecture','04 Prediction', "05 Business Prospects"])






if app_mode == '01 Introduction':
   
    introduction.Show_introduction()


elif app_mode == '03 Metrics and Model Architecture':
    data_viz.data_visualization(model_choice)

elif app_mode == '04 Prediction':
    prediction.Display_prediction(model_choice,label_dict)

elif app_mode == '02 Dataset visualization':
    dataset_viz.show_sample_images_page()


elif app_mode == "05 Business Prospects":


    st.title("💼 Business & Real-Life Applications of Facial Emotion Recognition")

    st.markdown("""
    Facial Emotion Recognition (FER) is transforming multiple industries by enabling systems to interpret and respond to human emotions. Below are real-world applications across sectors.

    ---

    ### 🛍️ Retail & Customer Experience
    - **Smart In-Store Cameras**: Detect real-time emotions to assess customer satisfaction, interest, or frustration.
    - **Product Testing**: Analyze emotional responses to new items or displays before launch.
    - **Personalized Marketing**: Deliver ads or recommendations based on detected mood.

    ---

    ### 🎮 Gaming & Entertainment
    - **Adaptive Gaming**: Change game difficulty, background music, or narrative pacing based on player emotions.
    - **Audience Analysis**: Capture real-time feedback on trailers, shows, or interactive experiences to tailor content.

    ---

    ### 🧠 Mental Health & Wellness
    - **Therapy Support**: Identify signs of emotional distress (e.g., anxiety, depression) during telehealth sessions.
    - **Mood Tracking Apps**: Use selfie inputs to monitor emotional patterns over time.
    - **Workplace Wellbeing**: Detect signs of stress or burnout in employees (with consent).

    ---

    ### 👨‍🏫 Education & e-Learning
    - **Engagement Monitoring**: Spot signs of boredom, confusion, or excitement in students during lessons.
    - **Teaching Feedback**: Use aggregated emotional data to refine teaching strategies and content delivery.

    ---

    ### 🤖 Human-Computer Interaction (HCI)
    - **Emotionally Aware Interfaces**: Improve AI assistants and robots with emotional adaptability.
    - **Driver Monitoring Systems**: Detect fatigue, anger, or distraction to improve safety in vehicles.

    ---

    ### 🔒 Security & Law Enforcement
    - **Threat Detection**: Monitor for abnormal stress or anger in sensitive areas.
    - **Interrogation Analysis**: Support analysts in reading non-verbal cues (must be ethically guided).

    ---

    ### 🎥 Market Research & Media Analytics
    - **Ad Testing**: Track emotional responses to advertisements for optimization.
    - **Political Campaigns**: Measure emotional impact of speeches and promotional content on viewers.

    ---

    ### ⚖️ Ethical Considerations
    - 🔒 **User Consent**: Always obtain explicit permission.
    - 🧠 **Bias Mitigation**: Ensure algorithms are trained across diverse populations.
    - 👁️‍🗨️ **Transparency**: Clearly communicate how data is used.
    - 🚫 **Avoid Surveillance Misuse**: Restrict usage to ethical, approved applications.

    ---
    """)

    with st.expander("🔧 Current Limitations & Future Improvements"):
        st.markdown("""
        While this app demonstrates the potential of facial emotion recognition, there are several areas for growth and refinement:

        1. **📉 Model Accuracy**
        - Current model accuracy, especially on the **test set**, needs improvement.
        - Training with more diverse and balanced data could help reduce overfitting and bias.

        2. **📦 Model Integration**
        - Loading the pre-trained **VGG model**  has been inconsistent.
        - Improvements in model saving/loading could enhance reliability.

        3. **🧠 Model Interpretability**
        - Implementing **Grad-CAM** or similar visualization tools would help explain predictions from both **ViT** and **VGG** models.
        - This is especially important for transparency and debugging.

        4. **🤝 Community Contributions**
        - Allowing users to **upload images with labels** could expand and diversify the dataset over time.
        - Consider building a secure, consent-based data submission pipeline.

        5. **🧪 Exploring New Architectures**
        - Testing alternative models such as **ResNet**, **EfficientNet**, or **Swin Transformers** could yield better performance.
        - Benchmarking across architectures can help select the best fit for emotion recognition tasks.

        ---
        These areas are actively being explored to make the application more accurate, interpretable, and scalable.
        """)

    st.info("This app outlines real-world use cases of facial emotion recognition across industries with emphasis on ethical deployment.")


else:
    st.write("Please select a valid option from the sidebar.")
