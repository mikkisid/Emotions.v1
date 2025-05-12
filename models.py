import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image




from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np





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



import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms





# Parameters
batch_size = 64




img_size = 100  # Updated from 48 to 100

# Transforms for CNN
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),           # Resize to 100x100
    transforms.RandomHorizontalFlip(),                 # Data augmentation
    transforms.RandomRotation(degrees=10),             # Data augmentation
    transforms.ToTensor(),                             # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




# Transforms for VGG and ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor [0,1]
    # transforms.RandomRotation(9),
    transforms.Normalize(               # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
        





# Then unzip it
import zipfile
import os

with zipfile.ZipFile('dataset_final.zip', 'r') as zip_ref:
    zip_ref.extractall('Dataset_final')


#datasets
train_dataset_cnn = datasets.ImageFolder(root='Dataset_final/train', transform=transform_train)
test_dataset_cnn = datasets.ImageFolder(root='Dataset_final/test', transform=transform_test)


train_dataset_v = datasets.ImageFolder(root='Dataset_final/train', transform=transform)
test_dataset_v = datasets.ImageFolder(root='Dataset_final/test', transform=transform)





# DataLoaders
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader_cnn = DataLoader(test_dataset_cnn,  batch_size=batch_size, shuffle=False, num_workers=2)


train_loader_v = DataLoader(train_dataset_v, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader_v = DataLoader(test_dataset_v, batch_size=batch_size, shuffle=False, num_workers=2)






























# -------------------------------
# Model: CNN (your custom model)
# -------------------------------
class FacialReaction(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialReaction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, padding=1)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)













class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Create a convolutional layer for patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # Flatten (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape

        # Linear transformation to get queries, keys and values
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        attention = torch.softmax(energy / (self.embed_dim ** 0.5), dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(B, N, E)
        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=768):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=768):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.layernorm1(x + attn_out)  # Add & Norm
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)  # Add & Norm
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192, num_heads=3, num_layers=12, num_classes=6):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # MLP Head for classification
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Embed the image into patches
        x = self.patch_embed(x)

        # Add class token to the sequence
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N_patches+1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head
        cls_output = x[:, 0]  # Extract the class token output
        out = self.fc_out(cls_output)

        return out








# -------------------------------
# Load Model Functions (correct filenames)
# -------------------------------
@st.cache_resource
def load_cnn_model():
    model = FacialReaction(num_classes=6)
    
    # Load full checkpoint
    checkpoint = torch.load('CNN_facial_reaction.pth',map_location='cpu')
    
    # Load only the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model

@st.cache_resource
def load_vgg_model():
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 6)
    model.load_state_dict(torch.load("vgg_dataset2_84_74.pth", map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_vit_model():
    model = VisionTransformer()

    model.heads = nn.Sequential(nn.Linear(192,6))


    checkpoint = torch.load('vit_70_67.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

                          
                          
    model.eval()
    return model

# -------------------------------
# Preprocess Webcam Image
# -------------------------------
def preprocess_image(img: Image.Image,model_type='CNN'):

    if model_type == 'CNN':
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(img).unsqueeze(0)  # [1, 3, 100, 100]

    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),      # Resize to 224x224
            transforms.ToTensor(),              # Convert to tensor [0,1]
            # transforms.RandomRotation(9),
            transforms.Normalize(               # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# ------------------------------------------
# ✂️ Tightly crop center of image
# ------------------------------------------
def tight_center_crop(img: Image.Image, crop_ratio: float = 0.7) -> Image.Image:
    """
    Crops a tighter square from the center of the image.
    crop_ratio defines the portion to keep (e.g., 0.7 means 70% of the smaller side).
    """
    width, height = img.size
    side = min(width, height)
    crop_size = int(side * crop_ratio)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return img.crop((left, top, right, bottom))



        













def apply_gradcam_streamlit(model, input_tensor, target_layer, class_names=None, true_label=None):
    """
    Applies Grad-CAM on a given image tensor and returns:
      - Original image
      - Original + Grad-CAM overlay

    Args:
    - model: Trained CNN/VGG/ViT model.
    - input_tensor: A single image tensor (1, 3, H, W).
    - target_layer: Target layer for Grad-CAM.
    - class_names: Optional dict mapping class indices to names.
    - true_label: Optional integer ground-truth label (1-indexed).

    Returns:
    - Tuple of original image and Grad-CAM overlay (both as NumPy arrays)
    """

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # GradCAM setup
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None

    # Run inference
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    predicted_label = predicted.item() + 1  # shift from 0–5 to 1–6

    # Grad-CAM computation
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]  # (H, W)

    # Unnormalize and prepare original image
    img_disp = input_tensor.squeeze(0).cpu()
    img_disp = img_disp * 0.5 + 0.5  # Assuming normalization was [-1, 1]
    img_disp = img_disp.permute(1, 2, 0).numpy()  # (H, W, C)

    # Create heatmap image
    heatmap_image = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)

    # Return both images for display
    return img_disp, heatmap_image, predicted_label

