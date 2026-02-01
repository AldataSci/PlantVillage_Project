# 🌿 Plant Disease Classification using Transfer Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://plantdiseaseclassifier-dutgkqlda6aoanj4hm8zxk.streamlit.app/)

An end-to-end computer vision application that identifies 15 different plant diseases using a fine-tuned ResNet18 architecture.

## 🚀 Live Demo
**[Click here to try the app!](https://plantdiseaseclassifier-dutgkqlda6aoanj4hm8zxk.streamlit.app/)**

## 📊 Results
- **Final Test Accuracy:** 98.45%
- **Validation Accuracy:** 98.51%
- **Baseline CNN Accuracy:** 93.8%

## 🧠 Project Narrative
In this project, I explored three different approaches to find the most effective model:
1. **Custom CNN:** Built a 3-layer CNN from scratch. Achieved a solid 93.8% accuracy.
2. **Transfer Learning (Frozen):** Used a pretrained ResNet18 but kept the backbone frozen. Accuracy dropped to 87.9%, proving that ImageNet features need adaptation for specialized botanical data.
3. **Fine-Tuning (Unfrozen):** Unfroze the ResNet18 backbone and trained with a low learning rate (1e-4). This achieved our peak performance of 98.45%.

### ⚠️ The "Domain Shift" Lesson
During deployment, I identified a performance gap when testing with "real-world" images from Google Search compared to the lab-controlled **PlantVillage** dataset. To address this, I implemented a **Sample Image** feature in the app to demonstrate the model's high performance on in-distribution data while providing transparency regarding out-of-distribution field images.

## 🛠️ Tech Stack
- **Framework:** PyTorch
- **Architecture:** ResNet18 (Fine-tuned)
- **Deployment:** Streamlit Cloud, Hugging Face (Weight Hosting)
- **Environment:** Google Colab (GPU Accelerated)
- **Libraries:** Torchvision, Scikit-Learn, PIL

## 📂 Data
The model was trained on a subset of the **PlantVillage** dataset, covering 15 classes including Tomato, Potato, and Pepper diseases. 
- **Training images:** 14,439
- **Validation images:** 3,097
- **Test images:** 3,101

## 📥 Pre-trained Model & Usage
Download the fine-tuned ResNet18 weights and class mapping from Hugging Face:
[🤗 Hugging Face Model Hub](https://huggingface.co/alihaq123/plant_diease_classifier/blob/main/plant_disease_resnet18.pth)

```python
# Load the weights (state_dict)
model.load_state_dict(torch.load('plant_disease_resnet18.pth', map_location=torch.device('cpu')))
model.eval()
