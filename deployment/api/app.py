from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.architectures import efficientnet 
import json
import io
import logging
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and configurations
try:
    logger.info("Loading model...")
    model = efficientnet.EfficientNet.from_name('efficientnet-b0')
    model.load_state_dict(torch.load("models/pretrained/efficientnet.onnx"))
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Model could not be loaded. Check model path or configurations.")

# Load model metadata for label decoding
try:
    with open('models/pretrained/model_metadata.json', 'r') as f:
        metadata = json.load(f)
        idx_to_class = metadata['idx_to_class']
    logger.info("Model metadata loaded successfully.")
except FileNotFoundError as fnf_error:
    logger.error(f"Metadata file not found: {fnf_error}")
    raise HTTPException(status_code=500, detail="Model metadata file is missing.")
except Exception as e:
    logger.error(f"Error loading model metadata: {e}")
    raise HTTPException(status_code=500, detail="An error occurred while loading model metadata.")

# Define image preprocessing function
def preprocess_image(image: Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# Utility function to handle image file upload and conversion
def read_image_file(file: UploadFile) -> Image.Image:
    try:
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid image.")

# Inference function to classify the image
def run_inference(image: torch.Tensor) -> str:
    image = image.unsqueeze(0)  # Add batch dimension
    try:
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            class_name = idx_to_class[str(class_idx)]
            return class_name
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

# API Endpoint to classify a single image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    logger.info(f"Received image for classification: {file.filename}")
    image = read_image_file(file)
    processed_image = preprocess_image(image)
    class_name = run_inference(processed_image)
    logger.info(f"Classification result for {file.filename}: {class_name}")
    return {"class_name": class_name}

# Batch image classification
@app.post("/classify_batch")
async def classify_images(files: List[UploadFile] = File(...)):
    logger.info(f"Received {len(files)} images for batch classification.")
    results = []
    for file in files:
        image = read_image_file(file)
        processed_image = preprocess_image(image)
        class_name = run_inference(processed_image)
        logger.info(f"Classification result for {file.filename}: {class_name}")
        results.append({"filename": file.filename, "class_name": class_name})
    return {"results": results}

# Health check endpoint to verify if the API is running
@app.get("/health")
async def health_check():
    logger.info("Health check request received.")
    return {"status": "API is running"}

# Exception handler for HTTP 400 errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error occurred: {exc.detail}")
    return {"error": exc.detail}

# Helper function for logging request details
def log_request_details(file: UploadFile):
    logger.info(f"Processing file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")
    logger.info(f"File size: {len(file.file.read())} bytes")
    file.file.seek(0)  # Reset file pointer after reading

# Helper function for validating image content types
def validate_image_file(file: UploadFile):
    valid_content_types = ["image/jpeg", "image/png"]
    if file.content_type not in valid_content_types:
        logger.error(f"Unsupported file format: {file.content_type}")
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    logger.info(f"File format validated: {file.content_type}")

# Another inference method with probabilistic outputs
def run_inference_with_probabilities(image: torch.Tensor):
    image = image.unsqueeze(0)
    try:
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            top5_class_names = [idx_to_class[str(idx.item())] for idx in top5_idx[0]]
            return top5_class_names, top5_prob[0].tolist()
    except Exception as e:
        logger.error(f"Inference with probabilities failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference with probabilities failed.")

# Endpoint for returning top 5 predictions with probabilities
@app.post("/classify_top5")
async def classify_image_top5(file: UploadFile = File(...)):
    logger.info(f"Received image for top-5 classification: {file.filename}")
    image = read_image_file(file)
    processed_image = preprocess_image(image)
    class_names, probabilities = run_inference_with_probabilities(processed_image)
    logger.info(f"Top-5 classification result for {file.filename}: {class_names}")
    return {"top_5_classes": class_names, "probabilities": probabilities}

# Error handling for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"An unexpected error occurred: {exc}")
    return {"error": "An unexpected error occurred. Please try again later."}

# Graceful server shutdown handling
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API gracefully.")

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)