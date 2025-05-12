import streamlit as st
import matplotlib.pyplot as plt


from torchinfo import summary

def data_visualization(model_choice):
    def plot_model_metrics(model_type):
        # Epochs are the same for all models
        epochs = list(range(1, 11))

        if model_type == 'CNN':
            loss = [
                1.2389, 0.9501, 0.8562, 0.7818, 0.7231, 0.6737, 0.6299, 0.5995, 0.5672, 0.5388,
                0.4650, 0.4469, 0.4263, 0.4116, 0.3860, 0.3775, 0.3621, 0.3455, 0.3250, 0.3098,
                0.2803, 0.2633, 0.2520, 0.2465, 0.2436, 0.2339, 0.2197, 0.2168, 0.2097, 0.2021
            ]

            accuracy = [
                53.36, 65.29, 69.07, 71.31, 73.55, 75.41, 77.23, 78.31, 79.55, 80.71,
                83.26, 84.22, 85.00, 85.37, 86.30, 86.78, 87.03, 87.60, 88.68, 89.18,
                90.26, 91.04, 91.20, 91.32, 91.74, 92.02, 92.82, 92.50, 93.00, 93.25
            ]

        elif model_type == 'VGG16':
            loss = [1.2832, 0.8841, 0.7730, 0.7002, 0.6222, 0.5854, 0.5632, 0.5135, 0.4946, 0.4537]
            accuracy = [56.87, 68.18, 72.45, 75.38, 78.16, 79.32, 80.08, 82.19, 82.61, 84.26]

        elif model_type == 'ViT':
            loss = [186.7186, 176.4275, 116.8164, 159.8890, 151.8824, 151.6594, 146.9743, 143.7478, 140.8833, 138.7943]
            accuracy = [63.57, 65.16, 66.85, 68.92, 70.29, 71.09, 71.87, 72.54, 73.11, 73.92]

        else:
            st.error("Model type must be one of: CNN, VGG16, ViT")
            return
        
        # Set epochs to match the loss list length
        epochs = list(range(1, len(loss) + 1))

        # Plot both Loss and Accuracy
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(epochs, loss, marker='o', color='tomato')
        axs[0].set_title(f"{model_type} - Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].grid(True)

        axs[1].plot(epochs, accuracy, marker='o', color='seagreen')
        axs[1].set_title(f"{model_type} - Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].grid(True)

        st.pyplot(fig)

    # Streamlit UI
    st.title("Model Training Metrics Viewer")
    plot_model_metrics(model_choice)



    st.subheader("📊 Model Architecture Summary")
    st.markdown("This section provides a detailed breakdown of the model architecture, including the number of parameters, trainability, and estimated model size.")

    if model_choice == "CNN":
        st.markdown("""
    #### 🤖 CNN Architecture (`FacialReaction`)
    | Layer              | Input Shape       | Output Shape      | Params     | Trainable |
    |-------------------|-------------------|-------------------|------------|-----------|
    | Conv2d (conv1)    | [1, 3, 100, 100]  | [1, 64, 99, 99]   | 3,136      | ✅        |
    | MaxPool2d         | [1, 64, 99, 99]   | [1, 64, 49, 49]   | -          | ❌        |
    | Conv2d (conv2)    | [1, 64, 49, 49]   | [1, 64, 48, 48]   | 65,600     | ✅        |
    | MaxPool2d         | [1, 64, 48, 48]   | [1, 64, 24, 24]   | -          | ❌        |
    | Linear (fc1)      | [1, 36864]        | [1, 128]          | 4,718,720  | ✅        |
    | Linear (fc2)      | [1, 128]          | [1, 6]            | 774        | ✅        |

    **Total Parameters**: `4,788,230`  
    **Trainable Parameters**: `4,788,230`  
    **Non-trainable Parameters**: `0`  
    **Estimated Model Size**: `~25.5 MB`  
    """)

    elif model_choice == "ViT":
        st.markdown("""
    #### 🧠 Vision Transformer (ViT) Architecture

    | Component                       | Input Shape        | Output Shape       | Params     | Trainable |
    |--------------------------------|--------------------|--------------------|------------|-----------|
    | Patch Embedding (Conv2d)       | [32, 3, 224, 224]  | [32, 192, 14, 14]  | 147,648    | ✅        |
    | Transformer Blocks (12x)       | [32, 197, 192]     | [32, 197, 192]     | ~5.3M      | ✅        |
    | Classification Head (fc_out)   | [32, 192]          | [32, 6]            | 1,158      | ✅        |

    **Total Parameters**: `5,526,348`  
    **Trainable Parameters**: `5,526,348`  
    **Non-trainable Parameters**: `0`  
    **Estimated Model Size**: `~1.3 GB`  
    """)

    elif model_choice == "VGG16":
        st.markdown("""
    #### 🏗️ VGG16 Architecture (Fine-Tuned)
    
    | Layer                         | Input Shape         | Output Shape        | Params       | Trainable |
    |------------------------------|---------------------|---------------------|--------------|-----------|
    | Features (13 conv layers)    | [32, 3, 224, 224]   | [32, 512, 7, 7]     | ~7.9M        | ❌        |
    | AdaptiveAvgPool2d            | [32, 512, 7, 7]     | [32, 512, 7, 7]     | 0            | ❌        |
    | Linear (fc1)                 | [32, 25088]         | [32, 4096]          | 102,764,544  | ✅        |
    | Linear (fc2)                 | [32, 4096]          | [32, 1024]          | 4,195,328    | ✅        |
    | Linear (fc3)                 | [32, 1024]          | [32, 6]             | 6,150        | ✅        |
    
    **Total Parameters**: `116,186,502`  
    **Trainable Parameters**: `106,966,022`  
    **Non-trainable Parameters**: `9,220,480`  
    **Estimated Model Size**: `~2.4 GB`
    """)
    
    else:
        st.warning("⚠️ Model summary not available.")
