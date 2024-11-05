from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import torch
import logging
from models import inference
from utils.data_loader import preprocess_image
from deployment.cpp_backend import cpp_inference
from deployment.model_metadata import get_model_info
from monitoring.logger import log_request, log_prediction

router = APIRouter()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImageClassificationAPI")

# Load model metadata and set up inference
model_metadata = get_model_info()
model_name = model_metadata.get("name")
model_path = model_metadata.get("path")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
logger.info(f"Loading model {model_name} from {model_path}")
model = inference.load_model(model_path, device)
logger.info("Model loaded successfully")


# Health check route
@router.get("/health", status_code=200)
async def health_check():
    logger.info("Health check request received")
    return JSONResponse(content={"status": "API is healthy", "model": model_name})


# Function to process and classify a single image
async def classify_single_image(file: UploadFile):
    try:
        # Read and process the image
        logger.info(f"Processing file: {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            cpp_result = cpp_inference(processed_image, model)
            predictions = inference.run_inference(model, processed_image, device)

        # Combine results from Python and C++ backends
        combined_predictions = inference.combine_results(cpp_result, predictions)

        return {"file": file.filename, "predictions": combined_predictions}

    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image {file.filename}: {str(e)}")


# Image classification route
@router.post("/classify/")
async def classify_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        result = await classify_single_image(file)

        # Log the request and prediction asynchronously
        background_tasks.add_task(log_request, file.filename, result)
        background_tasks.add_task(log_prediction, result)

        return JSONResponse(content=result)

    except HTTPException as e:
        logger.error(f"Classification failed: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Batch image classification route
@router.post("/batch_classify/")
async def batch_classify_images(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    try:
        batch_results = []
        logger.info(f"Processing batch of {len(files)} files")

        for file in files:
            result = await classify_single_image(file)
            batch_results.append(result)

            # Log each request and prediction asynchronously
            background_tasks.add_task(log_request, file.filename, result)
            background_tasks.add_task(log_prediction, result)

        return JSONResponse(content={"model": model_name, "batch_predictions": batch_results})

    except HTTPException as e:
        logger.error(f"Batch classification failed: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Additional image-related utilities for more robust handling

def validate_image_format(image: Image.Image):
    """
    Ensures the uploaded image is in an acceptable format.
    Only 'RGB' mode is allowed for classification tasks.
    """
    if image.mode != "RGB":
        logger.error("Invalid image format. Expected 'RGB'.")
        raise HTTPException(status_code=400, detail="Invalid image format. Expected 'RGB'.")


# Add route for image format validation
@router.post("/validate_image/")
async def validate_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Validating image format for {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        validate_image_format(image)

        return JSONResponse(content={"message": "Image format is valid", "file": file.filename})

    except Exception as e:
        logger.error(f"Error validating image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating image: {str(e)}")


# Add route for resizing image to meet input requirements
@router.post("/resize_image/")
async def resize_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Resizing image {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Resize to standard input size (224x224 for the model)
        resized_image = image.resize((224, 224))

        # Save resized image into memory for future processing
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG")
        buffer.seek(0)

        return JSONResponse(content={"message": "Image resized successfully", "file": file.filename})

    except Exception as e:
        logger.error(f"Error resizing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resizing image: {str(e)}")


# Handling unexpected exceptions at a global level
@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"An unexpected error occurred: {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred"})


# Utility function to preprocess and check images in bulk
async def preprocess_and_validate_images(files: List[UploadFile]):
    """
    This function preprocesses a batch of images and validates their formats.
    Useful for batch inference requests.
    """
    valid_images = []
    logger.info("Preprocessing and validating images in batch")

    for file in files:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            validate_image_format(image)
            processed_image = preprocess_image(image)
            valid_images.append(processed_image)

        except Exception as e:
            logger.error(f"Skipping file {file.filename}: {str(e)}")
            continue

    return valid_images


# Batch image preprocessing route
@router.post("/preprocess_batch/")
async def preprocess_batch_images(files: List[UploadFile] = File(...)):
    try:
        valid_images = await preprocess_and_validate_images(files)

        if not valid_images:
            raise HTTPException(status_code=400, detail="No valid images to process.")

        return JSONResponse(content={"message": "Batch preprocessing successful", "processed_count": len(valid_images)})

    except Exception as e:
        logger.error(f"Error during batch preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during batch preprocessing")