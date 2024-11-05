import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom block definitions for modular architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomCNN, self).__init__()
        
        # Layer definitions
        self.layer1 = ConvBlock(3, 64)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        
        # Additional layers for customization
        self.extra_conv1 = ConvBlock(256, 512)
        self.extra_conv2 = ConvBlock(512, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer1(x)  # First convolutional block
        x = self.layer2(x)  # Second convolutional block
        x = self.layer3(x)  # Third convolutional block

        x = self.extra_conv1(x)  # Additional convolution block 1
        x = self.extra_conv2(x)  # Additional convolution block 2
        
        # Apply Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialization function
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Utility function to initialize and get model
def get_custom_cnn(num_classes=1000, initialize_weights=True):
    model = CustomCNN(num_classes=num_classes)
    if initialize_weights:
        model.apply(init_weights)
    return model

# Advanced loss function using Label Smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

# Custom learning rate scheduler
def get_scheduler(optimizer, mode='cosine', base_lr=0.001, max_lr=0.01, step_size=2000):
    if mode == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    elif mode == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif mode == 'cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size)
    return scheduler

# Metrics module to evaluate the model
class MetricsTracker:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.num_batches = 0

    def update(self, outputs, labels, loss):
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        self.loss_sum += loss.item()
        self.num_batches += 1

    def accuracy(self):
        return 100 * self.correct / self.total

    def average_loss(self):
        return self.loss_sum / self.num_batches

# Training loop implementation
def train_model(model, dataloader, criterion, optimizer, scheduler=None, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        tracker = MetricsTracker()
        for inputs, labels in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            tracker.update(outputs, labels, loss)
        
        # Adjust learning rate if scheduler is used
        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Validation loop
def validate_model(model, dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    tracker = MetricsTracker()

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            tracker.update(outputs, labels, loss)

    print(f'Validation Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Advanced optimizer setups with gradient clipping
def configure_optimizer(model, learning_rate=0.001, weight_decay=1e-5, optimizer_type='adam'):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def gradient_clipping(optimizer, max_norm=1.0):
    for group in optimizer.param_groups:
        nn.utils.clip_grad_norm_(group['params'], max_norm)

# Utility function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)

# Utility function to load model checkpoints
def load_checkpoint(model, optimizer, filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']

# Early stopping mechanism to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def check(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Advanced data augmentation for the training pipeline
import torchvision.transforms as transforms

def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

# Custom dataset loader with optional data augmentation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_transforms, val_transforms = get_data_transforms()

    train_dataset = ImageFolder(root=f'{data_dir}/train', transform=train_transforms)
    val_dataset = ImageFolder(root=f'{data_dir}/val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# Visualization utilities for model performance metrics
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = len(train_losses)
    
    plt.figure(figsize=(14, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Training Loss", color='b')
    plt.plot(range(epochs), val_losses, label="Validation Loss", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accs, label="Training Accuracy", color='b')
    plt.plot(range(epochs), val_accs, label="Validation Accuracy", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.show()

# Custom learning rate finder
def find_lr(model, dataloader, optimizer, criterion, init_value=1e-8, final_value=10.0, beta=0.98):
    num = len(dataloader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []

    for inputs, labels in dataloader:
        batch_num += 1

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Track the best loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
    
    return log_lrs, losses

# Plotting learning rate vs loss to find optimal learning rate
def plot_lr_finder(log_lrs, losses):
    plt.plot(log_lrs, losses)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()

# Gradient accumulation for large batches
def train_with_accumulation(model, dataloader, criterion, optimizer, scheduler=None, accumulation_steps=4, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        tracker = MetricsTracker()
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            tracker.update(outputs, labels, loss)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Fine-tuning pretrained models with custom layers
from torchvision import models

class FineTunedCNN(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(FineTunedCNN, self).__init__()
        
        # Load a pretrained ResNet model
        self.base_model = models.resnet50(pretrained=pretrained)
        
        # Freeze the base layers if using pretrained weights
        if pretrained:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        # Additional layers for custom fine-tuning
        self.fc1 = nn.Linear(num_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to perform learning rate warmup
def lr_warmup(optimizer, warmup_steps, init_lr, final_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(warmup_steps - current_step) / float(max(1, warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

# Utility for fine-tuning a pretrained model
def fine_tune_model(model, dataloader, criterion, optimizer, scheduler=None, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        tracker = MetricsTracker()

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tracker.update(outputs, labels, loss)
        
        if scheduler:
            scheduler.step()

        print(f'Fine-tuning Epoch {epoch + 1}/{num_epochs}, Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Advanced evaluation with confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

# Model ensemble methods
class ModelEnsemble(nn.Module):
    def __init__(self, models_list, num_classes):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.fc = nn.Linear(len(models_list) * num_classes, num_classes)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Concatenate the outputs and pass through a final layer
        combined_output = torch.cat(outputs, dim=1)
        return self.fc(combined_output)

# Function to perform ensemble prediction
def ensemble_predict(ensemble_model, dataloader, criterion):
    ensemble_model.eval()
    tracker = MetricsTracker()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = ensemble_model(inputs)
            loss = criterion(outputs, labels)
            tracker.update(outputs, labels, loss)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Ensemble Validation Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')
    return all_labels, all_preds

# Multi-GPU training support for large-scale datasets
def setup_multi_gpu(model, device_ids):
    if torch.cuda.device_count() > 1:
        print(f'Using {len(device_ids)} GPUs for training.')
        model = nn.DataParallel(model, device_ids=device_ids)
    return model

# Mixed precision training for memory optimization
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader, criterion, optimizer, scheduler=None, scaler=None, num_epochs=25):
    if not scaler:
        scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        tracker = MetricsTracker()

        for inputs, labels in dataloader:
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tracker.update(outputs, labels, loss)

        if scheduler:
            scheduler.step()

        print(f'Mixed Precision Epoch {epoch + 1}/{num_epochs}, Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Advanced gradient accumulation with mixed precision support
def train_accum_mixed_precision(model, dataloader, criterion, optimizer, scheduler=None, accumulation_steps=4, scaler=None, num_epochs=25):
    if not scaler:
        scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        tracker = MetricsTracker()
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tracker.update(outputs, labels, loss)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Accumulated Loss: {tracker.average_loss():.4f}, Accuracy: {tracker.accuracy():.2f}%')

# Learning rate scheduler for large-scale datasets
def configure_lr_scheduler(optimizer, mode='warmup', warmup_steps=500, final_lr=0.01):
    if mode == 'warmup':
        scheduler = lr_warmup(optimizer, warmup_steps, init_lr=0.001, final_lr=final_lr)
    elif mode == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif mode == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    return scheduler

# Save model predictions and metrics to file
import json

def save_predictions_to_file(predictions, labels, filepath='predictions.json'):
    data = {'predictions': predictions, 'labels': labels}
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_predictions_from_file(filepath='predictions.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['predictions'], data['labels']

# Final utility: model summary display function
from torchsummary import summary

def display_model_summary(model, input_size=(3, 224, 224)):
    summary(model, input_size)

# Main training script combining all functionalities
def main_training_loop(model, train_loader, val_loader, num_classes, num_epochs=25, optimizer_type='adam', lr=0.001, weight_decay=1e-5):
    # Configure optimizer and loss function
    optimizer = configure_optimizer(model, learning_rate=lr, weight_decay=weight_decay, optimizer_type=optimizer_type)
    criterion = LabelSmoothingLoss(classes=num_classes)
    
    # Configure learning rate scheduler
    scheduler = configure_lr_scheduler(optimizer, mode='warmup')
    
    # Train and validate the model
    train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
    validate_model(model, val_loader, criterion)
    
    # Save model and optimizer state
    save_checkpoint(model, optimizer, num_epochs, 0, 0)  # Replace with actual loss and accuracy

# Usage of the main training loop
if __name__ == "__main__":
    model = get_custom_cnn(num_classes=1000)
    train_loader, val_loader = get_dataloaders(data_dir='data')
    main_training_loop(model, train_loader, val_loader, num_classes=1000, num_epochs=25)