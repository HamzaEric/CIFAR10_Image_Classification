# CIFAR-10 Image Classification: From MLPs to CNNs

![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## Overview
This project serves as an educational deep learning pipeline demonstrating the evolution of image classification architectures. It starts by exploring the limitations and unsuitability of standard Multi-Layer Perceptrons (MLPs) for spatial image data, and progressively transitions to Convolutional Neural Networks (CNNs), culminating in a custom deep network architecture.

## The Learning Journey: Why MLPs Fail on Images
1. **The Spatial Collapse:** Images are inherently two-dimensional spatial structures. Flattening a 32x32x3 RGB image into a 1D vector of 3,072 features for an MLP completely destroys spatial hierarchies, edge relationships, and textural context.
2. **Parameter Explosion:** Scaling MLPs for high-resolution images leads to a massive explosion in dense layer parameters, making them highly prone to overfitting and computationally prohibitive to train.
3. **The CNN Solution:** Convolutional Neural Networks resolve these bottlenecks by introducing local receptive fields (kernels) that slide across the image. This preserves spatial relationships and drastically reduces parameter counts through weight sharing and pooling.

## Project Structure

```text
CIFAR10_Image_Classification/
│
├── .venv/                                   # Virtual Environment (excluded from version control)
│
├── Models/
│   └── alexnet_lite_cifar10.pth             # Trained weights for the optimized CNN model
│
├── Notebooks/
│   ├── CIFAR_10_Image_classification.ipynb  # EDA, data loading, and the baseline MLP approach
│   ├── CNNs.ipynb                           # Introduction to convolutions and basic CNN structure
│   ├── Deeper_Networks.ipynb                # Implementing custom deep architectures (AlexNet Lite)
│   └── Advanced_Training.ipynb              # Optimization, data augmentation, and final evaluation
│
├── app.py                                   # Interactive web interface for model inference
├── requirements.txt                         # Project dependencies
└── README.md                                # Project documentation
```

## Getting Started

### Prerequisites
* Python 3.14 (or compatible)
* NVIDIA GPU with CUDA support (Recommended for training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HamzaEric/CIFAR10_Image_Classification.git
   cd CIFAR10_Image_Classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploring the Architecture Notebooks
Navigate to the `Notebooks/` directory to follow the architectural progression from MLPs to advanced CNNs. Start Jupyter Server:
```bash
jupyter notebook
```

### 2. Running the Inference App
The project includes a Streamlit front-end application to test the trained `alexnet_lite_cifar10.pth` model on custom images. Run the app via your terminal:
```bash
streamlit run app.py
```

## Dataset
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks), with 6,000 images per class. 

## Tech Stack
* **Deep Learning Framework:** PyTorch & Torchvision
* **Data Manipulation:** NumPy, Pandas
* **UI/Deployment:** Streamlit
* **Visualization:** Matplotlib
