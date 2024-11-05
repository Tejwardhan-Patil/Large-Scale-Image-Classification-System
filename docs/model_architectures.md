# Model Architectures

## Overview

This section provides an overview of the various model architectures used for image classification tasks in the system.

## Convolutional Neural Networks (CNN)

The `cnn.py` file contains the Python implementation of a standard Convolutional Neural Network (CNN). The model is suitable for basic image classification tasks.

- File: `models/architectures/cnn.py`
- Key Features:
  - Multiple convolutional layers
  - MaxPooling and Dropout for regularization
  - Softmax classifier at the output

## ResNet (Residual Networks)

The `resnet.py` file provides an implementation of the ResNet architecture, which utilizes residual blocks for deeper networks.

- File: `models/architectures/resnet.py`
- Key Features:
  - Residual blocks to avoid vanishing gradients
  - Depth options: ResNet-18, ResNet-50, etc.
  - Pretrained weights available: `models/pretrained/resnet50.pth`

## EfficientNet

The `efficientnet.py` file implements EfficientNet, a model known for balancing accuracy and computational efficiency.

- File: `models/architectures/efficientnet.py`
- Key Features:
  - Compound scaling of width, depth, and resolution
  - Available in variants: B0-B7
  - Pretrained weights: `models/pretrained/efficientnet.onnx`

## Custom Models

The `custom/` directory contains custom architectures designed for specific datasets or tasks.

- Files:
  - `custom_cnn.py`: Custom CNN for smaller datasets
  - `custom_resnet.py`: Custom ResNet variant
  - `custom_unet.py`: Custom UNet for segmentation tasks

## Performance Optimization

C++ implementations are available for performance-critical tasks such as custom neural network layers.

- File: `models/architectures/custom_layer.cpp`
