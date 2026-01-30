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

## ⚠️ Security Note
The `kaggle.json` API key used for data ingestion is **not** included in this repository for security reasons. To reproduce, please use your own Kaggle API credentials.
