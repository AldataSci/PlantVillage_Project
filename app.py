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
             
⚠️ **Note:** This model is a 'Lab Specialist' trained on the **PlantVillage** dataset. 
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

# 4. Demo Images Section
st.markdown("### Try These Sample Images")
st.markdown("Click a button below to test the model instantly:")

demo_images = {
    "Tomato Healthy": "https://raw.githubusercontent.com/AldataSci/plant_disease_classifier/refs/heads/main/samples_images/0031da2a-8edd-468f-a8b1-106657717a32___RS_HL%200105_Tomato_Healthy.JPG",
    "Tomato Spider Mites": "https://raw.githubusercontent.com/AldataSci/plant_disease_classifier/refs/heads/main/samples_images/003b7929-a364-4e74-be1c-37c4c0a6ec63___Com.G_SpM_FL%201414_Tomato_Spidermites.JPG",
    "Potato Early Blight": "https://raw.githubusercontent.com/AldataSci/plant_disease_classifier/refs/heads/main/samples_images/0267d4ca-522e-4ca0-b1a2-ce925e5b54a2___RS_Early.B%207020_Potato_Early_Blight.JPG"
}

cols = st.columns(len(demo_images))
for idx, (name, url) in enumerate(demo_images.items()):
    with cols[idx]:
        if st.button(f"Use {name}", key=name):
            st.session_state['demo_url'] = url
            st.session_state['demo_name'] = name
        st.image(url, caption=name, use_column_width=True)

st.markdown("---")

# 5. File Uploader or Demo Image Handler
uploaded_file = None
image = None

# Check if demo image was selected
if 'demo_url' in st.session_state:
    try:
        response = requests.get(st.session_state['demo_url'])
        image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(image, caption=f"Demo: {st.session_state.get('demo_name', 'Selected Image')}", width=400)
        uploaded_file = "demo"  # Trigger prediction flow
        
        # Add a button to clear demo and upload own image
        if st.button("Clear and Upload My Own Image"):
            del st.session_state['demo_url']
            if 'demo_name' in st.session_state:
                del st.session_state['demo_name']
            st.rerun()
    except Exception as e:
        st.error(f"Error loading demo image: {e}")
        del st.session_state['demo_url']
else:
    uploaded_file = st.file_uploader("Or upload your own leaf photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', width=400)

# 6. Prediction
if uploaded_file is not None and image is not None:
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
            st.info(f"**Confidence Score:** {confidence*100:.2f}%")

st.markdown("---")
st.caption("Developed as a Portfolio Project | Data Science & Deep Learning")
