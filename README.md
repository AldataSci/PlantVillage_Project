# Plant Disease Classification using Transfer Learning

This project uses a Deep Convolutional Neural Network (ResNet18) to identify 15 different types of plant diseases from the PlantVillage dataset.

## 🚀 Results
- **Final Test Accuracy:** 98.45%
- **Validation Accuracy:** 98.51%
- **Baseline CNN Accuracy:** 93.8%

## 🧠 Project Narrative
In this project, I explored three different approaches to find the most effective model:
1. **Custom CNN:** Built a 3-layer CNN from scratch. Achieved a solid 93.8% accuracy.
2. **Transfer Learning (Frozen):** Used a pretrained ResNet18 but kept the backbone frozen. Accuracy dropped to 87.9%, proving that ImageNet features need adaptation for specialized botanical data.
3. **Fine-Tuning (Unfrozen):** Unfroze the ResNet18 backbone and trained with a low learning rate (1e-4). This achieved our peak performance of 98.45%.

## 🛠️ Tech Stack
- **Framework:** PyTorch
- **Architecture:** ResNet18 (Fine-tuned)
- **Environment:** Google Colab (GPU Accelerated)
- **Libraries:** Torchvision, Scikit-Learn, Matplotlib, Seaborn

## 📊 Data
The model was trained on a subset of the **PlantVillage** dataset, covering 15 classes including Tomato, Potato, and Pepper diseases. 
- **Training images:** 14,439
- **Validation images:** 3,097
- **Test images:** 3,101

## 📈 Future Work
- Deploy the model using **Streamlit** for a web-based demo.
- Quantize the model for mobile deployment to assist farmers in low-connectivity areas.

## 📥 Pre-trained Model
Download the fine-tuned ResNet18 weights and class mapping from Hugging Face:
[🤗 Hugging Face Model Hub](https://huggingface.co/alihaq123/plant_diease_classifier/blob/main/plant_disease_resnet18.pth)

To use the model:
```python
import torch
import torch.nn as nn
from torchvision import models
import json

# 1. Rebuild the architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15) # 15 classes

# 2. Load the weights (state_dict)
# Ensure 'plant_disease_resnet18.pth' is in your directory
model.load_state_dict(torch.load('plant_disease_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# 3. Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print("Model loaded successfully!")
```

## ⚠️ Security Note
The `kaggle.json` API key used for data ingestion is **not** included in this repository for security reasons. To reproduce, please use your own Kaggle API credentials.
