# Data Pipeline

## Overview

The data pipeline involves the following stages: data collection, preprocessing, augmentation, and splitting the data into training, validation, and test sets.

## Data Preprocessing

Preprocessing includes resizing, normalization, and other steps to ensure uniformity in the dataset.

- File: `data/scripts/preprocess.py`
- Key Features:
  - Resizing images to a fixed dimension
  - Normalization of pixel values
  - Handling different image formats

## Data Augmentation

Augmentation is performed to improve the robustness of models by generating variations in the dataset.

- File: `data/scripts/augment.py`
- Augmentation Techniques:
  - Random rotations
  - Horizontal and vertical flipping
  - Zoom and translation transformations

## Data Splitting

The dataset is split into training, validation, and test sets to ensure proper evaluation.

- File: `data/scripts/split.py`
- Splitting Ratios:
  - 70% for training
  - 15% for validation
  - 15% for testing

## C++-based Preprocessing and Augmentation

For large datasets, C++ implementations are provided to improve performance.

- Files:
  - `data/scripts/cpp_preprocess.cpp`: High-performance preprocessing
  - `data/scripts/cpp_augment.cpp`: High-performance augmentation
