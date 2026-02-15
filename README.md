# Potato Leaf Disease Classification
### Hybrid ConvNeXt + Transformer Architecture

## 🌿 Project Overview
This project focuses on the early detection and classification of potato leaf diseases using deep learning. By identifying diseases like Early Blight and Late Blight accurately, farmers can take timely action to prevent significant crop loss.

The model utilizes a sophisticated **hybrid architecture** that combines Convolutional Neural Networks (CNNs) for local feature extraction with Transformers for global context understanding.

## 🏗️ Model Architecture
The classifier uses a custom hybrid approach built with TensorFlow and Keras:

* **Feature Extractor (CNN):** A pre-trained **ConvNeXtTiny** backbone (without the top layers) is used to capture fine spatial details and textures of leaf lesions.
* **Contextual Layer (Transformer):** A custom **Multi-Head Attention** encoder block follows the CNN. This allows the model to analyze global dependencies across different parts of the leaf image.
* **Classification Head:** A Multi-Layer Perceptron (MLP) consisting of Global Average Pooling, a Dropout layer (0.3) for regularization, and a Dense layer (128 units) leading to a Softmax output.

## 📊 Dataset & Performance
- **Source:** PlantVillage dataset.
- **Classes:** 3 (Early Blight, Late Blight, and Healthy).
- **Training Size:** 5,403 images.
- **Validation Size:** 763 images.

### Results after 20 Epochs:
| Metric | Value |
| :--- | :--- |
| **Training Accuracy** | ~99.6% |
| **Validation Accuracy** | **~97.4%** |
| **Test Loss** | 0.20 |

## 🛠️ Technologies Used
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Keras-CV, NumPy, Matplotlib
- **Platform:** Google Colab (GPU Accelerated)

## 🚀 How to Run
1. Clone this repository.
2. Open the `.ipynb` file in Google Colab.
3. Ensure you have the dependencies installed:
   ```bash
   pip install tensorflow keras-cv
