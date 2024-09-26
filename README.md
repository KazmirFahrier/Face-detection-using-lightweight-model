# Real-Time Face Recognition using MobileNetV2

This project implements a face recognition model using the **MobileNetV2** architecture, fine-tuned on a custom dataset of 5 individuals. The model is trained for real-time face recognition applications on embedded devices like Raspberry Pi but is initially developed and trained using **Kaggle** for testing and experimentation.

## Project Overview

- **Dataset**: The dataset consists of 25 images of faces belonging to 5 individuals. Data augmentation is applied to expand the dataset and improve generalization.
- **Model**: Pre-trained **MobileNetV2** is used as the backbone model, with additional layers added to adapt the model for the face recognition task. The model is trained to classify images into 5 classes (one for each individual).
- **Training**: The model is trained using **TensorFlow** and **Keras** on Kaggle. It incorporates techniques like data augmentation and **Early Stopping** to prevent overfitting.
- **Deployment**: The trained model is converted into **TensorFlow Lite** format for deployment on resource-constrained devices like the Raspberry Pi for real-time face recognition.

## Key Features

- **MobileNetV2**: Lightweight and optimized for embedded devices.
- **Data Augmentation**: Increases dataset variability with random flips, rotations, zooms, and brightness changes.
- **Early Stopping**: Automatically stops training when validation performance stops improving, reducing overfitting and saving computation time.
- **TensorFlow Lite**: Supports real-time face recognition on edge devices like the Raspberry Pi.

## How to Run

1. Train the model on Kaggle or any other environment using any similar structured dataset.
2. Convert the model to **TensorFlow Lite** for deployment on embedded systems.
3. Deploy the TensorFlow Lite model on Raspberry Pi or similar devices for real-time face recognition.

## Future Work

- Increase the dataset size for better generalization.
- Fine-tune the model with more classes for wider face recognition capabilities.
- Optimize the inference speed on embedded devices.

