import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.architectures import cnn, resnet, efficientnet  
from utils.data_loader import CustomDataset  
from utils.metrics import calculate_accuracy  
from utils.visualization import plot_loss_curve  
import json
import argparse
import yaml
import time

# Load configuration from YAML file
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Argument Parser for overriding configurations
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--batch_size', type=int, default=config['batch_size'], help="Batch size for training")
parser.add_argument('--learning_rate', type=float, default=config['learning_rate'], help="Learning rate for optimizer")
parser.add_argument('--epochs', type=int, default=config['epochs'], help="Number of training epochs")
parser.add_argument('--model', type=str, default=config['model'], help="Model architecture (cnn, resnet, efficientnet)")
parser.add_argument('--save_dir', type=str, default=config['save_dir'], help="Directory to save trained models")
parser.add_argument('--checkpoint', type=str, default=None, help="Path to a model checkpoint to resume training")
args = parser.parse_args()

# Device configuration: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations (for training set)
train_transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std']),
])

# Validation transformations (for validation set)
val_transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std']),
])

# Data Loading
print("Loading training data...")
train_dataset = CustomDataset(config['data']['train'], transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Loading validation data...")
val_dataset = CustomDataset(config['data']['val'], transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model selection based on argument
print(f"Selected model: {args.model}")
if args.model == "cnn":
    model = cnn.CNN().to(device)
elif args.model == "resnet":
    model = resnet.ResNet().to(device)
elif args.model == "efficientnet":
    model = efficientnet.EfficientNet().to(device)
else:
    raise ValueError(f"Model architecture {args.model} is not supported.")

# Load model from checkpoint if specified
if args.checkpoint:
    print(f"Loading model checkpoint from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

# Loss function and optimizer setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to validate the model on the validation dataset
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    val_loss /= len(val_loader)

    return val_loss, accuracy

# Function to train the model for one epoch
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % config['log_interval'] == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / (i + 1):.4f}, Accuracy: {100 * correct / total:.2f}%")

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1} completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    
    return running_loss / len(train_loader), 100 * correct / total

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_val_accuracy = 0.0
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        print(f"\n--- Starting epoch {epoch + 1}/{num_epochs} ---")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)

        # Validate after each epoch
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, os.path.join(args.save_dir, f"{args.model}_best.pth"))
            print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

    return training_losses, validation_losses, best_val_accuracy

# Function to log training and validation metrics to a file
def log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy, log_file):
    with open(log_file, 'a') as f:
        log_entry = (f"Epoch: {epoch+1}, "
                     f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")
        f.write(log_entry)
    print("Metrics logged to file.")

# Function to save the final model after training is completed
def save_final_model(model, save_dir, model_name):
    final_model_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

# Function to load the model from a saved checkpoint
def load_model_checkpoint(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

# Function to visualize the training and validation loss curves
def plot_training_curves(train_losses, val_losses, save_dir):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path)
    print(f"Loss curves saved to {plot_path}")
    plt.close()

# Function to visualize training and validation accuracy curves
def plot_accuracy_curves(train_accuracies, val_accuracies, save_dir):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(save_dir, 'accuracy_curves.png')
    plt.savefig(plot_path)
    print(f"Accuracy curves saved to {plot_path}")
    plt.close()

# Main training loop that orchestrates the entire training process
def main_training_loop():
    # Log file to track progress
    log_file = os.path.join(args.save_dir, 'training_log.txt')

    # Lists to track loss and accuracy for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    # Loop through each epoch
    for epoch in range(args.epochs):
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Log metrics
        log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy, log_file)

        # Append losses and accuracies to lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            save_model(model, best_model_path)
            print(f"Best model saved with accuracy: {best_val_accuracy:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, args.save_dir)
    plot_accuracy_curves(train_accuracies, val_accuracies, args.save_dir)

    # Save the final model after all epochs are done
    save_final_model(model, args.save_dir, args.model)

    # Print final results
    print(f"Training completed with best validation accuracy: {best_val_accuracy:.2f}%")

# Function to evaluate the trained model on the test dataset
def evaluate_on_test_set(test_loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy

# Setting up the test dataset and loader for final evaluation
def setup_test_evaluation():
    print("Loading test data...")
    test_dataset = CustomDataset(config['data']['test'], transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate on the test set
    test_loss, test_accuracy = evaluate_on_test_set(test_loader, model, criterion, device)

    # Log final test results
    with open(os.path.join(args.save_dir, 'final_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    print("Test results saved to final_test_results.txt.")

# Function to resume training from a checkpoint
def resume_training_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Resuming training from checkpoint {checkpoint_path}...")
    
    # Load the model from checkpoint
    load_model_checkpoint(model, checkpoint_path, device)

    # Continue with the training loop from the last epoch
    main_training_loop()

# Function to adjust hyperparameters dynamically
def dynamic_hyperparameter_adjustment(epoch):
    # Function to adjust learning rates or other parameters during training
    # Like reducing learning rate after a certain number of epochs
    if epoch % 10 == 0:
        new_lr = args.learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate adjusted to {new_lr} at epoch {epoch+1}")

# Function to set up experiment logging
def setup_experiment_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'experiment_log.txt')
    print(f"Experiment log will be saved to {log_file}")
    return log_file

# Function to perform early stopping if validation loss stops improving
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Function to handle multiple model checkpoints during training
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

# Function to load checkpoint and resume training
def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"Checkpoint {path} not found. Starting from scratch.")
        return 0

# Function to visualize filters of the first convolutional layer
def visualize_conv_filters(layer, save_dir):
    filters = layer.weight.data.cpu().numpy()
    import matplotlib.pyplot as plt
    num_filters = filters.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(filters[i, 0, :, :], cmap='gray')
        ax.axis('off')
    plt.savefig(os.path.join(save_dir, 'conv_filters.png'))
    plt.close()
    print(f"Convolutional filters saved to {save_dir}/conv_filters.png")

# Function to adjust batch size dynamically if out-of-memory error occurs
def dynamic_batch_size_adjustment(train_loader, current_batch_size, adjust_factor=0.5):
    print(f"Adjusting batch size from {current_batch_size}")
    new_batch_size = int(current_batch_size * adjust_factor)
    print(f"New batch size: {new_batch_size}")
    
    # Create new DataLoader with reduced batch size
    train_loader = DataLoader(train_loader.dataset, batch_size=new_batch_size, shuffle=True, num_workers=4)
    return train_loader, new_batch_size

# Function to handle model pruning (optional feature)
def prune_model(model, pruning_percentage=0.2):
    from torch.nn.utils import prune
    print(f"Pruning {pruning_percentage * 100}% of the model weights")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
            print(f"Pruned {name}")
    print("Model pruning complete.")

# Function to visualize feature maps after the first convolutional layer
def visualize_feature_maps(model, data_loader, save_dir):
    model.eval()
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    import matplotlib.pyplot as plt
    with torch.no_grad():
        # Forward pass to the first conv layer
        conv1 = model.conv1(images)
    
    feature_maps = conv1.cpu().numpy()
    fig, axes = plt.subplots(1, feature_maps.shape[1], figsize=(20, 20))
    
    for i, ax in enumerate(axes):
        ax.imshow(feature_maps[0, i, :, :], cmap='gray')
        ax.axis('off')
    
    plt.savefig(os.path.join(save_dir, 'feature_maps.png'))
    plt.close()
    print(f"Feature maps saved to {save_dir}/feature_maps.png")

# Main entry point for the training script
if __name__ == "__main__":
    print("Starting training process...")

    # Set up experiment logging
    experiment_log = setup_experiment_logging(args.save_dir)

    # Load the model from a checkpoint if available
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping']['patience'], 
                                   min_delta=config['early_stopping']['min_delta'])

    # Training loop with early stopping
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)

        # Validate after each epoch
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Log metrics
        log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy, experiment_log)

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, os.path.join(args.save_dir, f"{args.model}_best.pth"))
            print(f"Best model saved with accuracy: {best_val_accuracy:.2f}%")

        # Save checkpoints at regular intervals
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Perform early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Exiting training loop.")
            break

        # Adjusting learning rate
        scheduler.step()

    # Save the final model after training
    save_final_model(model, args.save_dir, args.model)

    # Test the model on the test dataset
    setup_test_evaluation()

    print("Training and evaluation completed.")