import os
import json
import argparse
import logging
import yaml
import time
import torch
import shutil
from models import cnn, resnet, efficientnet
from utils import data_loader, metrics, visualization
from train import train_model
from monitoring import send_alert

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from YAML file
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

# Model selection logic based on config
def select_model(config):
    model_type = config['model']['type']
    if model_type == 'cnn':
        logger.info("Initializing CNN model")
        return cnn.CNN(**config['model']['params'])
    elif model_type == 'resnet':
        logger.info("Initializing ResNet model")
        return resnet.ResNet(**config['model']['params'])
    elif model_type == 'efficientnet':
        logger.info("Initializing EfficientNet model")
        return efficientnet.EfficientNet(**config['model']['params'])
    else:
        raise ValueError(f"Unknown model type specified: {model_type}")

# Save model checkpoints during training
def save_checkpoint(model, epoch, save_path, filename='checkpoint.pth'):
    filepath = os.path.join(save_path, filename)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, filepath)
    logger.info(f"Checkpoint saved at epoch {epoch} to {filepath}")

# Load model checkpoint
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

# Run the experiment with the loaded configuration
def run_experiment(config_path):
    start_time = time.time()
    logger.info("Starting experiment")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup output directories
    save_path = config['output']['model_save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.info(f"Model save path: {save_path}")
    
    # Load dataset
    try:
        train_loader, val_loader, test_loader = data_loader.load_data(config['data'])
        logger.info("Data loaders created successfully")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        send_alert("Data loading failed")
        return
    
    # Select and initialize model
    try:
        model = select_model(config)
    except ValueError as ve:
        logger.error(ve)
        send_alert(f"Model initialization failed: {ve}")
        return
    
    # Check if a checkpoint is provided
    if 'checkpoint' in config['output']:
        checkpoint_path = config['output']['checkpoint']
        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(model, checkpoint_path)
        else:
            logger.warning(f"Checkpoint path not found: {checkpoint_path}, starting from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training and evaluation process
    try:
        history = train_model(
            model,
            train_loader,
            val_loader,
            config['training']['epochs'],
            config['training']['optimizer'],
            config['training']['scheduler'],
            config['training']['loss_function'],
            config['training']['device'],
            start_epoch=start_epoch,
            early_stopping=config['training'].get('early_stopping', False)
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        send_alert(f"Training failed: {e}")
        return

    # Evaluate model on test set
    try:
        test_metrics = metrics.evaluate(model, test_loader, config['training']['device'])
        logger.info(f"Test set metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        send_alert(f"Evaluation failed: {e}")
        return
    
    # Save the trained model and metrics
    try:
        model_save_path = os.path.join(save_path, 'final_model.pth')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        metrics_save_path = os.path.join(save_path, 'metrics.json')
        with open(metrics_save_path, 'w') as f:
            json.dump(test_metrics, f)
        logger.info(f"Metrics saved to {metrics_save_path}")
    except Exception as e:
        logger.error(f"Failed to save model or metrics: {e}")
        send_alert(f"Saving failed: {e}")
    
    # Check if visualization is enabled
    if config['output'].get('visualize', False):
        try:
            visualization.plot_training_history(history)
            logger.info("Training history visualized")
        except Exception as e:
            logger.error(f"Failed to visualize training history: {e}")
    
    # Logging total time taken
    total_time = time.time() - start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds")

    # Clean up temporary files if any
    temp_dir = config['output'].get('temp_dir', None)
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary files cleaned up from {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary files: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a machine learning experiment with a given configuration.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.config)
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        send_alert(f"Experiment failed: {e}")