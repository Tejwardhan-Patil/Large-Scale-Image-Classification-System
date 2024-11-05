import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

# Define the CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(CNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Max-Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max-pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 512 * 7 * 7)
        
        # Apply fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Define a Custom Dataset Class for Image Classification
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load the images and labels from the data directory
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

# Helper Function to Load an Image
def load_image(img_path):
    from PIL import Image
    image = Image.open(img_path).convert('RGB')
    return image

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate Dataset and Dataloader
data_dir = 'data/processed'
dataset = ImageDataset(data_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Initialize the CNN Model
model = CNN(num_classes=1000)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# Function to Calculate Accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Update running loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy

    avg_loss = running_loss / len(val_loader)
    avg_accuracy = running_accuracy / len(val_loader)
    return avg_loss, avg_accuracy

# Training and Validation Loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_accuracy = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Update running loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, epoch, best_accuracy, 'best_model.pth')

# Function to Save the Model Checkpoint
def save_model(model, epoch, accuracy, filepath):
    print(f'Saving model at epoch {epoch} with accuracy {accuracy:.4f}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy
    }, filepath)

# Function to Load the Model Checkpoint
def load_model(filepath, model, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")

# Instantiate a Validation Dataset and DataLoader
val_data_dir = 'data/validation'
val_dataset = ImageDataset(data_dir=val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Begin Training and Validation
train_and_validate(
    model=model,
    train_loader=dataloader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device
)

# Function to Perform Inference on a Single Image
def predict_image(model, image_path, transform, device):
    model.eval()  # Set the model to evaluation mode
    image = load_image(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# Usage of the Inference Function
test_image_path = 'data/test/some_image.jpg'
predicted_label = predict_image(model, test_image_path, transform, device)
print(f'Predicted Label for the Test Image: {predicted_label}')

# Scheduler to Adjust Learning Rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Loop with Learning Rate Scheduler
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, labels)

        running_loss += loss.item()
        running_accuracy += accuracy

    scheduler.step()  # Step the learning rate scheduler

    avg_train_loss = running_loss / len(dataloader)
    avg_train_accuracy = running_accuracy / len(dataloader)

    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Data Augmentation Techniques
advanced_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmented Dataset and DataLoader
augmented_dataset = ImageDataset(data_dir=data_dir, transform=advanced_transform)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, num_workers=4)

# Logging Setup for Training
class TrainingLogger:
    def __init__(self):
        self.epoch_logs = []

    def log_epoch(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        self.epoch_logs.append(log_entry)
        self._print_log(log_entry)

    def _print_log(self, log_entry):
        print(f"Epoch {log_entry['epoch']} - "
              f"Train Loss: {log_entry['train_loss']:.4f}, "
              f"Train Accuracy: {log_entry['train_accuracy']:.4f}, "
              f"Val Loss: {log_entry['val_loss']:.4f}, "
              f"Val Accuracy: {log_entry['val_accuracy']:.4f}")

    def save_logs(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.epoch_logs, f, indent=4)
        print(f"Training logs saved to {filename}")

# Instantiate the Logger
logger = TrainingLogger()

# Training with Logger
def train_with_logging(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, logger, scheduler=None):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_accuracy = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Update running loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy

        # Step the learning rate scheduler, if provided
        if scheduler:
            scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Log the epoch results
        logger.log_epoch(epoch+1, avg_train_loss, avg_train_accuracy, val_loss, val_accuracy)

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, epoch, best_accuracy, 'best_model_with_logs.pth')

# Start Training with Advanced Augmentation and Logging
train_with_logging(
    model=model,
    train_loader=augmented_dataloader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    logger=logger,
    scheduler=scheduler
)

# Confusion Matrix Calculation
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.nditer(np.array([[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Generate Confusion Matrix
def evaluate_model_on_test_data(model, test_loader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')

# Test Dataset and DataLoader
test_data_dir = 'data/test'
test_dataset = ImageDataset(data_dir=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Class names are provided in a list 'class_names'
class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# Evaluate the Model on Test Data
evaluate_model_on_test_data(model, test_loader, class_names, device)

# Save the Logs After Training
logger.save_logs('training_logs.json')

# Function to Load and Visualize Training Logs
def load_and_visualize_logs(log_file):
    import json
    with open(log_file, 'r') as f:
        logs = json.load(f)

    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train_loss'] for log in logs]
    val_losses = [log['val_loss'] for log in logs]
    train_accuracies = [log['train_accuracy'] for log in logs]
    val_accuracies = [log['val_accuracy'] for log in logs]

    # Plotting Loss Curves
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy Curves
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Loading and Visualizing Logs
load_and_visualize_logs('training_logs.json')