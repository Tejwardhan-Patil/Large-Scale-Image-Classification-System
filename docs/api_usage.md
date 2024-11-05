# API Usage

## Overview

The REST API is designed to serve the trained image classification model in production environments.

## Endpoints

### `/predict`

This endpoint accepts an image and returns classification predictions.

- Method: `POST`
- Payload: Image file
- Response: JSON object with predicted labels and probabilities

### `/health`

The health endpoint is used to check the status of the API.

- Method: `GET`
- Response: `200 OK` if the service is running

## API Implementation

### Python Backend

The API backend is implemented in Python using Flask or FastAPI.

- File: `deployment/api/app.py`

### C++ Backend for Performance

For low-latency applications, a C++ backend is provided.

- File: `deployment/api/cpp_backend.cpp`
- Features:
  - Optimized for high-throughput, low-latency inference
