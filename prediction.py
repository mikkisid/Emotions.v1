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



import models



def Display_prediction(model_choice, label_dict,):
    # Camera input
    img_file = st.camera_input("📸 Take a photo to classify")
    # Load the selected model
    if model_choice == "CNN":
        model = models.load_cnn_model()
    elif model_choice == "VGG16":
        model = models.load_vgg_model()
    else:
        model = models.load_vit_model()
    



    if img_file is not None:
        image = Image.open(img_file)

        # 🔍 Tightly crop the center to focus on the face
        cropped_image = models.tight_center_crop(image, crop_ratio=0.7)

        # Show cropped image to user
        st.image(cropped_image, caption="🧠 Tightly Center-Cropped Image")


        predict = False
        if st.button("🧠 Predict Emotion"):
            predict = True
        if predict:
            st.write("🧠 Predicting...")

            input_tensor = models.preprocess_image(cropped_image, model_type=model_choice)


            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item() + 1  

            st.success(f"🧠 Predicted Emotion: **{label_dict[predicted_label]}**")


            if model_choice == "CNN":
                target_layer = model.conv2  # Adjust to your CNN

                # Grad-CAM
                orig, gradcam_img, pred_label = models.apply_gradcam_streamlit(
                    model=model,
                    input_tensor=input_tensor,
                    target_layer=target_layer,
                    class_names=label_dict,
                    true_label=None
                )

                st.subheader("🧠 Grad-CAM Visualization")



                # Convert both images to PIL
                orig_img_pil = Image.fromarray((orig * 255).astype(np.uint8))
                heatmap_img_pil = Image.fromarray(gradcam_img)

                # Side-by-side view
                st.image([orig_img_pil, heatmap_img_pil], caption=["Original", "Grad-CAM"], width=300)




    if st.button("🎲 Show Random Prediction From Test Dataset"):
        model.eval()

        test_dataset = models.test_dataset_cnn





        if model_choice != "CNN":
            test_dataset = models.test_dataset_v

        # Pick a truly random image from the whole dataset


        index_to_label = {i: int(cls) for i, cls in enumerate(test_dataset.classes)}  # test_dataset.classes should be strings like ['1', '2', ..., '6']


        total_samples = len(test_dataset)
        rand_index = random.randint(0, total_samples - 1)

        # Load image and label directly
        image, label = test_dataset[rand_index]
        input_tensor = image.unsqueeze(0)  # Add batch dimension

        # Run prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        # Convert class index (0-based) to folder label (1-based)
        true_label = int(test_dataset.classes[label])
        predicted_label = int(test_dataset.classes[predicted.item()])


        

        # Convert image for display
        image_disp = image.permute(1, 2, 0).cpu().numpy()
        image_disp = image_disp * 0.5 + 0.5  # unnormalize
        image_disp = np.clip(image_disp, 0, 1)

        # Display image using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_disp)
        ax.set_title(f"✅ True: {label_dict[true_label]}\n🤖 Predicted: {label_dict[predicted_label]}")
        ax.axis("off")
        st.pyplot(fig)





        if model_choice == "CNN":
            # ----------------------------
            # 🧠 Apply Grad-CAM on Selected Random Image
            # ----------------------------

        
            target_layer = model.conv2

            # Prepare the single image tensor for Grad-CAM
            input_tensor = image.unsqueeze(0)

            # Grad-CAM
            img_disp, gradcam_overlay, _ = models.apply_gradcam_streamlit(
                model=model,
                input_tensor=input_tensor,
                target_layer=target_layer,
                class_names=label_dict,
                true_label=true_label
            )

            st.subheader("🔥 Grad-CAM on Random Test Image")

            # Convert both to displayable format
            orig_pil = Image.fromarray((img_disp * 255).astype(np.uint8))
            heatmap_pil = Image.fromarray(gradcam_overlay)

            # Side-by-side in Streamlit
            st.image([orig_pil, heatmap_pil], caption=["Original", "Grad-CAM"], width=300)
