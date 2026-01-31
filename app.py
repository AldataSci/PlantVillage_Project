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
st.markdown("""
This app uses a **Fine-Tuned ResNet18** model to identify 15 different plant diseases.
             
 **Note:** This model is a 'Lab Specialist' trained on the **PlantVillage** dataset. 
It performs best on **close-up photos of single leaves** against **neutral, plain backgrounds**.
""")

# 2. Load Model & Metadata
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    
    weights_url = "https://huggingface.co/alihaq123/plant_diease_classifier/resolve/38ddadd9f2cffbd655627ade1778ab72cf805524/plant_disease_resnet18.pth"
    response = requests.get(weights_url)
    model.load_state_dict(torch.load(BytesIO(response.content), map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_labels():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_labels()

# 3. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. UI - File Uploader
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=400)
    
    if st.button('Identify Disease'):
        with st.spinner('Analyzing leaf patterns...'):
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probs, 0)
            
            label = class_names[str(predicted.item())]
            
            # Display Results
            st.success(f"**Prediction:** {label.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')}")
            st.progress(float(confidence))
            st.info(f"**Confidence Score:** {confidence100:.2f}%")

st.markdown("---")
st.caption("Developed as a Portfolio Project | Data Science & Deep Learning")
        with st.spinner('Analyzing leaf pattern...'):
            # Preprocess and Predict
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)

                # ===== DEBUG OUTPUT (ADD THIS) =====
                #st.write("raw outputs (first 5):", outputs[0][:5])
                probs = torch.softmax(outputs, dim=1)[0]
                topk = torch.topk(probs, k=5)
                #st.write("Top-5 indices:", topk.indices.tolist())
                #st.write("Top-5 probs:", [float(x) for x in topk.values])
                # ===== END DEBUG OUTPUT =====
                
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
            
            # Display Results
            label = class_names[str(predicted.item())]
            st.success(f"**Prediction:** {label}")
            st.progress(float(confidence))
            st.info(f"**Confidence:** {confidence*100:.2f}%")
st.markdown("---")
st.caption("Developed as a Portfolio Project | Data Science & Deep Learning")
