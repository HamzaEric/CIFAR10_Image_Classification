import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import plotly.express as px
import os

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(layout="wide")

MODEL_PATH = "Models/alexnet_lite_cifar10.pth"
IMG_SIZE = 32  # Architecture expects 32x32 input

# CIFAR-10 Classes
CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Create Model directory if it doesn't exist (for demonstration if file is missing)
os.makedirs("Models", exist_ok=True)

# ==========================================
# 2. Model Architecture Definition
# ==========================================
class AlexNetMiniLite(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Input: [B, 3, 32, 32]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# 3. Model Loading & Preprocessing
# ==========================================
@st.cache_resource
def load_model():
    """Instantiates model, loads weights, sets to evaluation mode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetMiniLite(num_classes=len(CLASSES))

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`. Please ensure the file exists.")
        # Return dummy model to prevent crash, though inference will be random
        return model.to(device), device

    try:
        # Load weights, handling CPU/GPU mapping automatically
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

@st.cache_resource
def get_transform():
    """Defines the preprocessing pipeline matching CIFAR-10 training stats."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Standard CIFAR-10 normalization constants
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

# Load resources
with st.spinner("Loading optimized classification model..."):
    model, device = load_model()
    transform_pipeline = get_transform()

# ==========================================
# 4. Session State & Callbacks
# ==========================================
# Initialize session state for the active image URL
if 'active_image_url' not in st.session_state:
    st.session_state.active_image_url = None

def set_sample_image(url):
    """Callback to update session state when a 'Try This' button is clicked."""
    st.session_state.active_image_url = url
    # Clear uploaded file if a sample is selected
    if 'file_uploader_key' in st.session_state:
        st.session_state['file_uploader_key'] += 1

# ==========================================
# 5. UI Layout - Header
# ==========================================
st.title("CIFAR-10 Image Classification")
st.markdown("""
Predict the category of an image using a custom-trained, optimized **AlexNetMiniLite** architecture.
This model distinguishes between 10 standard object classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.
""")
st.markdown("---")

# ==========================================
# 6. UI Layout - Input Section
# ==========================================
st.subheader("Select Input Image")

# Sample Images Section adapted for CIFAR-10 classes
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

# We use accessible public domain thumbnails for demonstration
sample_plane = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdPLeMiWjQCh964ibZVPLWACRjuj3ZPqXgc2KWqIL4qg&s=10"
sample_car = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZABlHvfCriar1Ugpol_2YinYJHf4p9pjEebpCfl59Xg&s=10"
sample_cat = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6vKrzE7RJpKf7dCmUxswuwIZpw_jIs7TtILmcufOzdA&s=10"
sample_ship = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9C3VxfBEPGJVDJ6ARnqg6DY7Ag7MOiw0HjN2YQxiSlA&s=10"

with col_s1:
    st.image(sample_plane, caption="Airplane", width=128)
    st.button("Try This", key="btn_plane", on_click=set_sample_image, args=(sample_plane,))

with col_s2:
    st.image(sample_car, caption="Automobile", width=128)
    st.button("Try This", key="btn_car", on_click=set_sample_image, args=(sample_car,))

with col_s3:
    st.image(sample_cat, caption="Cat", width=128)
    st.button("Try This", key="btn_cat", on_click=set_sample_image, args=(sample_cat,))

with col_s4:
    st.image(sample_ship, caption="Ship", width=128)
    st.button("Try This", key="btn_ship", on_click=set_sample_image, args=(sample_ship,))

st.info("Select a sample or upload your own. Scroll down to see predictive analytics.")
st.markdown("---")

# File uploader fallback
# Use a dynamic key to clear uploader programmatically
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

uploaded_file = st.file_uploader(
    "Or upload your own image (it will be resized to 32x32)...",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state['file_uploader_key']}"
)

# ==========================================
# 7. Image Resolution & Display Logic
# ==========================================
# Resolve active image source
active_image = None

# Preference 1: User Uploaded File
if uploaded_file is not None:
    active_image = Image.open(uploaded_file).convert("RGB")
    # Reset active URL if file is uploaded
    st.session_state.active_image_url = None

# Preference 2: Session State Sample URL
elif st.session_state.active_image_url is not None:
    try:
        # Emulate User-Agent for GStatic images
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(st.session_state.active_image_url, headers=headers, timeout=5)
        response.raise_for_status()
        active_image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load sample image: {e}")
        st.session_state.active_image_url = None # Reset state on failure

# ==========================================
# 8. Inference & Results Section
# ==========================================
if active_image is not None and model is not None:
    st.markdown("---")
    st.subheader("Predictive Analytics")

    # Side-by-side layout: Left column for image, Right column for prediction & graph
    col_img, col_res = st.columns([1, 1.2])

    with col_img:
        st.image(active_image, caption="Selected Input Image", use_container_width=True)
        predict_clicked = st.button("Classify Image", type="primary")

    with col_res:
        if predict_clicked:
            with st.spinner("Analyzing visual features (Conv -> ReLU -> Pool)..."):
                # Preprocess image
                transformed = transform_pipeline(active_image).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(transformed)
                    # Apply Softmax to get probabilities
                    probabilities = F.softmax(outputs, dim=1)[0]
                    predicted_idx = torch.argmax(probabilities).item()

                predicted_class = CLASSES[predicted_idx]
                confidence_score = probabilities[predicted_idx].item() * 100

            # Results Section
            st.success(f"**Top Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence_score:.2f}%")

            # Interactive Plotly Breakdown Chart
            st.subheader("Confidence Breakdown across Classes")

            # Create DataFrame for Plotly
            df_probs = pd.DataFrame({
                "Object Class": CLASSES,
                "Confidence (%)": [probabilities[i].item() * 100 for i in range(len(CLASSES))]
            }).sort_values(by="Confidence (%)", ascending=True)

            # Create Horizontal Bar Chart
            fig = px.bar(
                df_probs,
                x="Confidence (%)",
                y="Object Class",
                orientation="h",
                text_auto=".1f",
                color="Confidence (%)",
                # Use Viridis scale as specified in template
                color_continuous_scale="Viridis"
            )

            # Style the chart
            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="",
                xaxis=dict(range=[0, 105]), # Small padding at end
                coloraxis_showscale=False, # Hide the color legend scale
                margin=dict(l=0, r=0, t=20, b=0),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Welcome state before clicking predict
            st.info("Click **Classify Image** to run the AlexNetMiniLite evaluation model on this input.")

elif model is None:
    st.warning("Model is not loaded. Ensure weights exist at the specified path.")