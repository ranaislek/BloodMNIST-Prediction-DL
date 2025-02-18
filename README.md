### **Machine Learning for Human Data 2024/25 - Blood Cell Type Prediction Using Deep Learning**
#### **Accurate Classification of Blood Cell Types with CNN, ResNet, VGG-16, and Inception-v3**
##### Authors: Rana Islek, Enxhi Nushi
##### Date: February 2025

<div align="center">
    <img src="https://decision-for-liver.eu/wp-content/uploads/2020/07/UNIPD.png" alt="UniPd Logo" width="300"/>
</div>
---

![Blood Cells](https://raw.githubusercontent.com/your-username/your-repo/main/cells.png)  

*Microscopic images of different blood cell types from the BloodMNIST dataset.*

---

## **📌 Overview**
This project explores deep learning-based classification of blood cell types using the **BloodMNIST** dataset, which consists of **17,092** microscopic images of eight blood cell types. The goal is to enhance automated **hematological diagnostics** by evaluating multiple deep learning architectures, including:
- **Convolutional Neural Networks (CNN)**
- **ResNet**
- **VGG-16**
- **Inception-v3**
- **Attention mechanisms (SE & CBAM)**

### 🔬 **Key Research Questions**
1. Can lightweight models achieve competitive classification performance?
2. Do attention mechanisms (Squeeze-and-Excitation & CBAM) improve accuracy?
3. What is the trade-off between model complexity, accuracy, and computational efficiency?

---

## **📂 Dataset**
The **BloodMNIST** dataset is part of the **MedMNIST** collection and includes:
- **8 blood cell types** (basophils, eosinophils, erythroblasts, lymphocytes, monocytes, neutrophils, immature granulocytes, and platelets)
- **64×64 RGB microscopic images**
- **Pre-split into train (11,959), validation (1,712), and test (3,421) sets**

📌 **Note:** The dataset is publicly available via **[MedMNIST](https://medmnist.com/)**.

---

## **🛠️ Methods and Techniques**
### **1️⃣ Processing Pipeline**
- **Data Preprocessing:** Normalization, resizing, and augmentation
- **Model Architectures:** CNN, ResNet, Inception-v3, and VGG-16
- **Regularization Techniques:** Dropout, batch normalization, L2 weight regularization
- **Attention Mechanisms:** Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM)
- **Optimization:** Adam & SGD optimizers with categorical cross-entropy loss

### **2️⃣ Deep Learning Models**
| Model         | Test Accuracy (%) |
|--------------|----------------|
| **VGG-16**  | **96.43** |
| ResNet      | 95.67 |
| CNN + SE    | 95.03 |
| CNN + Batch Norm & Dropout | 95.59 |
| Inception-v3 | 88.02 |
| Basic CNN (4 layers) | 88.78 |

**Best Model:** **VGG-16** with **dropout layers**, achieving **96.43% accuracy**.

---

## **📊 Performance Analysis**
### **✔️ Accuracy vs. Computational Cost**
- **VGG-16** achieves the highest accuracy but requires **high computational power**.
- **CNN with batch normalization** provides a balance between accuracy (**95.59%**) and efficiency.
- **Inception-v3** struggles with **small datasets** like BloodMNIST.

### **⚠️ Trade-offs Considered**
- **Accuracy:** VGG-16 (96.43%) > ResNet (95.67%) > CNN (95.59%)
- **Execution Speed:** CNN models are **faster** but **less accurate**.
- **Memory Consumption:** ResNet has the **highest** memory footprint.

### **📈 Confusion Matrix (VGG-16)**
The best model (**VGG-16**) shows excellent classification across all 8 cell types, with minimal misclassification.

---

## **📌 How to Use**
### **🔧 Requirements**
- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV (for image processing)

---

## **📚 Future Work**
- Implement **lightweight models** for mobile deployment.
- Train with **larger medical datasets** for better generalization.
- Experiment with **transformer-based architectures** like Vision Transformers (ViTs).

---

## **📜 Citation**
If you use this project in your research, please cite:
```
@article{BloodMNIST,
  title={Blood Cell Type Prediction Using Deep Learning},
  author={Islek Nushi},
  journal={Deep Learning for Medical Imaging},
  year={2025}
}
```
