import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO

# 1. Setup Page Config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a photo of a plant leaf to identify the disease.")

# 2. Load Model & Metadata (Cached so it only runs once)
@st.cache_resource
def load_model():
    # Rebuild the ResNet18 Architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15) # Your 15 classes
    
    # Download weights from your Hugging Face
    # Replace this URL with your actual "Raw" link from Hugging Face
    weights_url = "https://huggingface.co/alihaq123/plant_diease_classifier/resolve/38ddadd9f2cffbd655627ade1778ab72cf805524/plant_disease_resnet18.pth"
    response = requests.get(weights_url)

    st.write("HF status:", response.status_code)
    st.write("HF content-type:", response.headers.get("content-type"))
    st.write("HF bytes:", len(response.content))
    # ===== END DEBUG =====
    
    model.load_state_dict(torch.load(BytesIO(response.content), map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_labels():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_labels()

## Debug Panel 

st.subheader("Debug (temporary)")

# Confirm class_names type and a few entries
st.write("class_names type:", type(class_names))
if isinstance(class_names, dict):
    st.write("class_names keys sample:", list(class_names.keys())[:5])
    st.write("class_names values sample:", [class_names[k] for k in list(class_names.keys())[:5]])
else:
    st.write("class_names sample:", class_names[:5])

# Confirm model final layer shape
st.write("model.fc:", model.fc)

# Confirm weights look non-trivial
w = model.fc.weight.detach().cpu()
st.write("fc weight mean/std:", float(w.mean()), float(w.std()))

## End Debug panel

# 3. Image Preprocessing (Must match your training transforms!)
preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. UI - File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict Button
    if st.button('Identify Disease'):
        with st.spinner('Analyzing...'):
            # Preprocess and Predict
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)

                # ===== DEBUG OUTPUT (ADD THIS) =====
                st.write("raw outputs (first 5):", outputs[0][:5])
                probs = torch.softmax(outputs, dim=1)[0]
                topk = torch.topk(probs, k=5)
                st.write("Top-5 indices:", topk.indices.tolist())
                st.write("Top-5 probs:", [float(x) for x in topk.values])
                # ===== END DEBUG OUTPUT =====
                
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
            
            # Display Results
            label = class_names[str(predicted.item())]
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")
