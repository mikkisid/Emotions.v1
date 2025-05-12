import streamlit as st
from PIL import Image


def Show_introduction():
    # Set page configuration
   

    # Load and display the image
    image = Image.open("emo.jpg")

    # Center the image using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image)



    # Title and subtitle
    st.title("🎭 Facial Emotion Recognition")
    st.subheader("Detecting Emotions from Facial Expressions Using Deep Learning")

    # Introduction text
    st.markdown("""
    Welcome to the **Face Emotion Recognition** app!  
    This project demonstrates the use of deep learning to recognize human emotions from facial expressions in real time.

    Using a convolutional neural network (CNN) trained on facial image datasets, the model can classify emotions such as **Happy**, **Sad**, **Angry**, **Surprised**, and more.
    """)

    # Add a separator
    st.markdown("---")

    # Motivation section
    st.header("💡 Motivation")
    st.markdown("""
    Facial expressions are a fundamental mode of non-verbal communication.  
    With the rise of AI and human-computer interaction, emotion recognition has gained importance in applications such as:
    - Mental health monitoring
    - Customer feedback analysis
    - Security and surveillance systems
    - Interactive gaming and virtual assistants
    """)

    # Objective section
    st.header("🎯 Objective")
    st.markdown("""
    The goal of this project is to:
    - Build a robust deep learning model that can accurately classify emotions from facial images.
    - Deploy the model in a user-friendly interface for real-time predictions.
    - Explore how AI can understand human affect through facial features.
    """)

    # How it works section
    st.header("⚙️ How It Works")
    st.markdown("""
    1. Upload an image or use your webcam to capture a face.
    2. The model detects the face and analyzes facial features.
    3. It then predicts the most likely emotion and displays the result.

    This app was built with **Streamlit**, and **PyTorch**.
    """)

    # Model Overview section
    st.header("🧠 Models Used")

    # Dropdown for model selection
    model_choice = st.selectbox(
        "Select a model to learn more about it:",
        ["Convolutional Neural Network (CNN)", "Vision Transformer (ViT)", "VGG"]
    )

    if model_choice == "Convolutional Neural Network (CNN)":
        st.subheader("🌀 Convolutional Neural Network (CNN)")
        cnn_image = Image.open("Convolutional-Neural-Network.jpg")  # Replace with your actual image file
        st.image(cnn_image, caption="Typical CNN architecture")
        st.markdown("""
        CNNs are specialized deep learning models for image processing.  
        They consist of layers that automatically learn to detect features like edges, textures, and patterns in images.

        ### 📍 Where It's Used:
        - **Face recognition systems** (e.g., in mobile phones)
        - **Medical imaging** (e.g., detecting tumors)
        - **Autonomous vehicles** (e.g., recognizing road signs and pedestrians)

        In our project, CNNs serve as a baseline for detecting emotions from faces due to their efficiency and interpretability. They are especially good when dealing with relatively smaller datasets.
        """)

    elif model_choice == "Vision Transformer (ViT)":
        st.subheader("🧠 Vision Transformer (ViT)")
        vit_image = Image.open("vit.jpg")  # Replace with your actual image file
        st.image(vit_image, caption="Vision Transformer concept")
        st.markdown("""
        ViTs bring the power of transformer models to the vision domain by splitting images into patches and processing them using self-attention — a technique originally used in NLP.

        ### 📍 Where It's Used:
        - **Large-scale image classification** (e.g., ImageNet tasks)
        - **Fine-grained object detection**
        - **Art analysis and medical diagnosis**

        In our app, ViT is used for capturing global relationships in facial features that might not be easily detected by CNNs. It's especially effective with high-resolution images and large training sets.
        """)

    elif model_choice == "VGG":
        st.subheader("🏗️ VGG Network")
        vgg_image = Image.open("new41.jpg")  # Replace with your actual image file
        st.image(vgg_image, caption="VGG architecture overview")
        st.markdown("""
        The VGG model, introduced by the Visual Geometry Group at Oxford, is known for its deep yet simple architecture using small (3x3) convolution filters.

        ### 📍 Where It's Used:
        - **Facial recognition systems**
        - **Emotion detection**
        - **Transfer learning tasks**, where VGG is pre-trained on large datasets like ImageNet and fine-tuned for specific applications.

        We use VGG as a benchmark in our system. While it's more computationally intensive than CNN, it performs well when high accuracy is prioritized over speed.
        """)

    # Footer or next step
    st.markdown("---")
    st.info("👉 Use the sidebar to get started and test the model with your own images or webcam.")