# Image Classification with CNNs on CIFAR-10

This repository contains implementations of various Convolutional Neural Network (CNN) architectures for image classification on the CIFAR-10 dataset. The project demonstrates the use of custom CNNs, pre-trained models (VGG16 and ResNet50), and transfer learning techniques to achieve high accuracy in classifying images into 10 categories.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Architectures Implemented](#architectures-implemented)  
4. [Requirements](#requirements)   
5. [Results](#results)  
6. [Acknowledgments](#acknowledgments)

---

## Introduction
The CIFAR-10 dataset is a widely used benchmark for evaluating computer vision models. This project explores:
- Custom CNN implementation.
- Transfer learning with **VGG16** and **ResNet50**.
- Performance evaluation and comparison between models.

The project focuses on understanding the impact of model complexity and transfer learning on image classification accuracy.

---

## Dataset
The **CIFAR-10** dataset consists of:
- **Training Set**: 50,000 images.
- **Test Set**: 10,000 images.
- Each image has dimensions of 32x32x3 (RGB) and belongs to one of 10 classes:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

---

## Architectures Implemented
1. **Custom CNN**:
   - 3 convolutional layers with increasing filter sizes.
   - MaxPooling, Flattening, and Dense layers.
   - Optimized with the Adam optimizer and Sparse Categorical Crossentropy.

2. **VGG16**:
   - Transfer learning using the pre-trained VGG16 model.
   - Added custom dense layers for classification.

3. **ResNet50**:
   - Transfer learning using the pre-trained ResNet50 model.
   - Adapted for CIFAR-10 classification using upscaling and additional dense layers.

---

## Requirements
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- Keras


---



### Run the Notebook
1. Launch Jupyter Notebook or any compatible IDE.
2. Open `cnn-cifar10-transferlearning.ipynb`.
3. Execute cells to train and evaluate the models.

### Example Results Visualization
The notebook includes:
- Model accuracy and loss plots.
- Sample image predictions with ground truth.

---

## Results
- **Custom CNN**:
  - Validation Accuracy: ~70%
- **VGG16**:
  - Validation Accuracy: ~85%
- **ResNet50**:
  - Validation Accuracy: ~90%

Accuracy may vary based on hyperparameters and training setup.

---

## Acknowledgments
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Pre-trained Models](https://keras.io/api/applications/)

Feel free to use this repository as a reference or extend it for more advanced projects!

--- 
