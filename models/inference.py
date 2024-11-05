import torch
import tensorflow as tf
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import os
import logging
import time

# Set up logging to track inference execution details
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_type, model_path, device='cpu', input_size=(224, 224)):
        self.model_type = model_type.lower()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        logger.info(f"Initializing inference engine for {self.model_type} model.")
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if self.model_type == 'cnn' or self.model_type == 'resnet':
            logger.info(f"Loading PyTorch model from {model_path}.")
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        elif self.model_type == 'efficientnet':
            logger.info(f"Loading TensorFlow model from {model_path}.")
            return tf.keras.models.load_model(model_path)
        elif self.model_type == 'onnx':
            logger.info(f"Loading ONNX model from {model_path}.")
            return ort.InferenceSession(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Preprocessing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        preprocess_pipeline = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess_pipeline(image).unsqueeze(0)
    
    def postprocess_output(self, output):
        logger.info(f"Postprocessing output: {output}")
        if self.model_type in ['cnn', 'resnet']:
            prediction = torch.argmax(output, dim=1).cpu().numpy()
        elif self.model_type == 'efficientnet':
            prediction = np.argmax(output, axis=1)
        elif self.model_type == 'onnx':
            prediction = np.argmax(output[0], axis=1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return prediction
    
    def predict(self, image_path):
        start_time = time.time()
        logger.info(f"Starting prediction for image: {image_path}")
        
        input_image = self.preprocess_image(image_path)
        
        if self.model_type in ['cnn', 'resnet']:
            input_image = input_image.to(self.device)
            with torch.no_grad():
                output = self.model(input_image)
            prediction = self.postprocess_output(output)
        elif self.model_type == 'efficientnet':
            input_image = np.expand_dims(input_image.numpy(), axis=0)
            output = self.model.predict(input_image)
            prediction = self.postprocess_output(output)
        elif self.model_type == 'onnx':
            input_image = input_image.numpy()
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_image})
            prediction = self.postprocess_output(output)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds.")
        return prediction, inference_time

    def batch_predict(self, image_dir):
        if not os.path.isdir(image_dir):
            raise ValueError(f"{image_dir} is not a valid directory.")
        
        logger.info(f"Running batch inference on images in directory: {image_dir}")
        predictions = {}
        total_time = 0.0
        images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_name in images:
            image_path = os.path.join(image_dir, image_name)
            prediction, inference_time = self.predict(image_path)
            predictions[image_name] = prediction
            total_time += inference_time
            logger.info(f"Image {image_name} predicted in {inference_time:.2f} seconds.")
        
        logger.info(f"Batch inference completed in {total_time:.2f} seconds.")
        return predictions, total_time

if __name__ == "__main__":
    # Load model metadata for label mapping
    with open("models/pretrained/model_metadata.json", 'r') as f:
        label_map = json.load(f)

    # Run inference using ResNet model
    model_inference = ModelInference(model_type="resnet", model_path="models/pretrained/resnet50.pth", device='cuda')
    
    # Single image inference
    image_path = "data/sample_image.jpg"
    prediction, inference_time = model_inference.predict(image_path)
    
    # Log prediction result
    logger.info(f"Predicted label: {label_map[str(prediction[0])]}, Time: {inference_time:.2f}s")

    # Batch inference
    image_dir = "data/batch_images/"
    batch_predictions, total_time = model_inference.batch_predict(image_dir)
    
    # Log batch inference result
    logger.info(f"Batch prediction completed. Total time: {total_time:.2f}s")

    # Define an inference method that can handle multiple input sizes
    def predict_with_dynamic_input(self, image_path, input_size):
        logger.info(f"Predicting with dynamic input size: {input_size}")
        self.input_size = input_size
        return self.predict(image_path)

    # Save predictions to a JSON file for record keeping
    def save_predictions(self, predictions, output_file="predictions.json"):
        logger.info(f"Saving predictions to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(predictions, f)
    
    # Load predictions from a JSON file
    def load_predictions(self, prediction_file):
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
        
        logger.info(f"Loading predictions from {prediction_file}")
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        return predictions

    # Extended batch processing to handle images with multiple resolutions
    def batch_predict_with_dynamic_input(self, image_dir, input_sizes):
        if not os.path.isdir(image_dir):
            raise ValueError(f"{image_dir} is not a valid directory.")
        
        if len(input_sizes) == 0:
            raise ValueError("Input sizes list cannot be empty.")
        
        logger.info(f"Running batch inference with dynamic input sizes on images in {image_dir}")
        predictions = {}
        total_time = 0.0
        images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, image_name in enumerate(images):
            image_path = os.path.join(image_dir, image_name)
            input_size = input_sizes[idx % len(input_sizes)]
            prediction, inference_time = self.predict_with_dynamic_input(image_path, input_size)
            predictions[image_name] = prediction
            total_time += inference_time
            logger.info(f"Image {image_name} predicted in {inference_time:.2f} seconds with input size {input_size}.")
        
        logger.info(f"Dynamic batch inference completed in {total_time:.2f} seconds.")
        return predictions, total_time

    # Provide a method to analyze prediction performance
    def analyze_performance(self, predictions, ground_truth, label_map):
        logger.info("Analyzing performance...")
        correct = 0
        total = len(predictions)
        for image_name, predicted_label in predictions.items():
            actual_label = ground_truth.get(image_name)
            predicted_label_name = label_map[str(predicted_label[0])]
            if predicted_label_name == actual_label:
                correct += 1
            else:
                logger.info(f"Misclassified: {image_name} - Predicted: {predicted_label_name}, Actual: {actual_label}")
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Accuracy: {accuracy:.2f}")
        return accuracy

    # Evaluation method for a test dataset directory
    def evaluate_on_test_set(self, test_dir, ground_truth_file, input_sizes):
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        logger.info(f"Evaluating model on test set located in {test_dir}")
        
        # Load ground truth labels
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        predictions, total_time = self.batch_predict_with_dynamic_input(test_dir, input_sizes)
        
        accuracy = self.analyze_performance(predictions, ground_truth, label_map)
        logger.info(f"Evaluation completed with accuracy: {accuracy:.2f}, Total inference time: {total_time:.2f} seconds")
        return accuracy, total_time

    # Method to convert a model to ONNX for performance optimization
    def convert_to_onnx(self, output_file, input_shape=(1, 3, 224, 224)):
        if self.model_type not in ['cnn', 'resnet']:
            raise ValueError(f"Model conversion to ONNX only supported for PyTorch models, not {self.model_type}.")
        
        logger.info(f"Converting model to ONNX format. Output file: {output_file}")
        
        sample_input = torch.randn(input_shape).to(self.device)
        torch.onnx.export(self.model, sample_input, output_file, opset_version=11, 
                          input_names=['input'], output_names=['output'])
        logger.info(f"Model successfully converted to {output_file}")
    
    # Function to benchmark model inference speed
    def benchmark_inference_speed(self, image_path, num_trials=10):
        logger.info(f"Benchmarking inference speed for {image_path} with {num_trials} trials.")
        
        times = []
        for _ in range(num_trials):
            start_time = time.time()
            self.predict(image_path)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        logger.info(f"Average inference time over {num_trials} trials: {avg_time:.4f} seconds.")
        return avg_time
    
    # Method to export predictions and performance metrics to a CSV file
    def export_to_csv(self, predictions, output_file="predictions.csv"):
        logger.info(f"Exporting predictions and performance metrics to {output_file}")
        with open(output_file, 'w') as f:
            f.write("Image,Predicted_Label,Inference_Time\n")
            for image_name, prediction in predictions.items():
                f.write(f"{image_name},{prediction[0]},\n")

    # Function to integrate with external APIs for remote inference
    def integrate_with_api(self, image_path, api_url, api_key):
        logger.info(f"Sending image {image_path} for inference to API at {api_url}")
        image = open(image_path, 'rb').read()
        import requests
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.post(api_url, files={'image': image}, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API Inference result: {result}")
            return result
        else:
            logger.error(f"API request failed with status code {response.status_code}")
            return None
    
    # Model loading for TensorFlow SavedModel format
    def load_savedmodel(self, model_dir):
        logger.info(f"Loading TensorFlow SavedModel from directory: {model_dir}")
        return tf.saved_model.load(model_dir)

    # Run inference using the TensorFlow SavedModel
    def predict_with_savedmodel(self, image_path, saved_model_dir):
        model = self.load_savedmodel(saved_model_dir)
        input_image = self.preprocess_image(image_path)
        logger.info(f"Running inference with TensorFlow SavedModel on {image_path}")
        
        input_image = tf.convert_to_tensor(input_image.numpy())
        output = model(input_image)
        prediction = np.argmax(output, axis=1)
        logger.info(f"Predicted label using SavedModel: {prediction}")
        return prediction

    # Method to handle multi-model ensemble predictions
    def ensemble_predict(self, image_path, models_info):
        """
        Perform inference using an ensemble of models.
        Args:
            image_path (str): Path to the image for prediction.
            models_info (list): List of dictionaries containing model_type and model_path.
        Returns:
            Final prediction from the ensemble of models.
        """
        logger.info("Running ensemble prediction using multiple models.")
        
        ensemble_predictions = []
        for model_info in models_info:
            model_type = model_info.get('model_type')
            model_path = model_info.get('model_path')
            
            # Initialize inference for each model in the ensemble
            model_inference = ModelInference(model_type=model_type, model_path=model_path, device=self.device)
            prediction, _ = model_inference.predict(image_path)
            ensemble_predictions.append(prediction)
        
        # Use majority voting for the final ensemble decision
        final_prediction = self.majority_voting(ensemble_predictions)
        logger.info(f"Final ensemble prediction: {final_prediction}")
        return final_prediction

    # Helper method to apply majority voting for ensemble predictions
    def majority_voting(self, predictions):
        logger.info("Applying majority voting on ensemble predictions.")
        predictions = np.array(predictions).flatten()
        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        return final_prediction

    # Function to visualize prediction results using matplotlib
    def visualize_predictions(self, image_path, prediction, label_map):
        """
        Visualizes the prediction result by displaying the image and predicted label.
        Args:
            image_path (str): Path to the input image.
            prediction (int): Predicted label index.
            label_map (dict): Dictionary mapping label indices to label names.
        """
        import matplotlib.pyplot as plt
        
        logger.info(f"Visualizing prediction for {image_path}")
        image = Image.open(image_path).convert('RGB')
        predicted_label = label_map.get(str(prediction[0]), "Unknown")

        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

    # Extended function to support multi-output model predictions
    def multi_output_predict(self, image_path, model_path):
        """
        Perform inference for multi-output models (e.g., segmentation + classification).
        Args:
            image_path (str): Path to the input image.
            model_path (str): Path to the multi-output model.
        Returns:
            Dictionary containing predictions for each output.
        """
        logger.info(f"Performing multi-output prediction with model: {model_path}")
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        input_image = self.preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            outputs = model(input_image)
        
        predictions = {f"output_{i}": torch.argmax(output, dim=1).cpu().numpy() for i, output in enumerate(outputs)}
        logger.info(f"Multi-output prediction results: {predictions}")
        return predictions

    # Method to optimize model loading and inference with TorchScript
    def convert_to_torchscript(self, output_file, input_shape=(1, 3, 224, 224)):
        """
        Converts the PyTorch model to TorchScript format for optimized inference.
        Args:
            output_file (str): Path to save the TorchScript model.
            input_shape (tuple): Shape of the input tensor.
        """
        if self.model_type not in ['cnn', 'resnet']:
            raise ValueError(f"TorchScript conversion only supported for PyTorch models, not {self.model_type}.")
        
        logger.info(f"Converting model to TorchScript format. Output file: {output_file}")
        
        sample_input = torch.randn(input_shape).to(self.device)
        scripted_model = torch.jit.trace(self.model, sample_input)
        scripted_model.save(output_file)
        logger.info(f"Model successfully converted to TorchScript and saved at {output_file}")

    # Function to measure memory usage during inference
    def memory_usage_benchmark(self, image_path):
        """
        Measures memory usage during model inference.
        Args:
            image_path (str): Path to the input image.
        """
        import tracemalloc
        
        logger.info("Starting memory usage benchmark.")
        tracemalloc.start()
        
        self.predict(image_path)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Current memory usage: {current / 10**6:.2f} MB; Peak memory usage: {peak / 10**6:.2f} MB")

    # Support for exporting multi-output models to ONNX
    def export_multi_output_onnx(self, model_path, output_file, input_shape=(1, 3, 224, 224)):
        """
        Exports a multi-output PyTorch model to ONNX format.
        Args:
            model_path (str): Path to the PyTorch multi-output model.
            output_file (str): Path to save the ONNX model.
            input_shape (tuple): Shape of the input tensor.
        """
        logger.info(f"Exporting multi-output model to ONNX format: {output_file}")
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        sample_input = torch.randn(input_shape).to(self.device)
        torch.onnx.export(model, sample_input, output_file, opset_version=11, 
                          input_names=['input'], output_names=['output_1', 'output_2'])
        logger.info(f"Multi-output model successfully exported to {output_file}")

    # Function to handle model quantization for optimized inference
    def quantize_model(self, model_path, output_file):
        """
        Applies post-training quantization to the model for optimized inference.
        Args:
            model_path (str): Path to the model to be quantized.
            output_file (str): Path to save the quantized model.
        """
        logger.info(f"Starting model quantization for {model_path}")
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Fuse model layers before quantization
        logger.info("Fusing model layers for quantization.")
        model.fuse_model()  
        
        # Apply static quantization
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
        # Save quantized model
        torch.save(model, output_file)
        logger.info(f"Quantized model saved at {output_file}")

    # Method to integrate with cloud storage for saving/loading models and results
    def integrate_with_cloud_storage(self, model_path, cloud_url, api_key):
        """
        Upload model to cloud storage or download from cloud storage.
        Args:
            model_path (str): Path to the model file.
            cloud_url (str): Cloud storage URL for upload/download.
            api_key (str): API key for authentication with cloud storage service.
        """
        import requests
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Uploading model {model_path} to cloud storage at {cloud_url}")
        with open(model_path, 'rb') as model_file:
            response = requests.post(cloud_url, files={'file': model_file}, headers={'Authorization': f'Bearer {api_key}'})
        
        if response.status_code == 200:
            logger.info(f"Model successfully uploaded to cloud storage.")
        else:
            logger.error(f"Failed to upload model to cloud. Status code: {response.status_code}")