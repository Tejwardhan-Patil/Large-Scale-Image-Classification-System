import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

# URL for pretrained EfficientNet weights
model_urls = {
    'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
    'efficientnet_b1': 'https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth'
}

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

# Mobile Inverted Bottleneck Convolution (MBConv) Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)

        # Depthwise Convolution
        self.dwconv = nn.Conv2d(
            mid_channels, mid_channels, kernel_size, stride=stride, 
            padding=kernel_size // 2, groups=mid_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Squeeze-and-Excitation block
        self.se = SEBlock(mid_channels)

        # Projection phase
        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection condition
        self.use_residual = (in_channels == out_channels and stride == 1)

        # Activation function
        self.activation = Swish()

    def forward(self, x):
        residual = x

        # Expansion phase
        if self.expand_ratio != 1:
            x = self.activation(self.bn0(self.expand_conv(x)))

        # Depthwise convolution and SE block
        x = self.activation(self.bn1(self.dwconv(x)))
        x = self.se(x)

        # Projection phase
        x = self.bn2(self.project_conv(x))

        # Residual connection
        if self.use_residual:
            x = x + residual
        return x

# EfficientNet Model Definition
class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()

        base_channels = 32  # Base number of channels
        final_channels = 1280  # Final number of channels before classifier

        # Stem convolution layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            Swish()
        )

        # EfficientNet block configurations (expand_ratio, kernel_size, stride, out_channels, num_blocks)
        block_params = [
            (1, 3, 1, 16, 1),   # Block 1
            (6, 3, 2, 24, 2),   # Block 2
            (6, 5, 2, 40, 2),   # Block 3
            (6, 3, 2, 80, 3),   # Block 4
            (6, 5, 1, 112, 3),  # Block 5
            (6, 5, 2, 192, 4),  # Block 6
            (6, 3, 1, 320, 1)   # Block 7
        ]

        self.blocks = nn.ModuleList([])  # Container for blocks
        in_channels = base_channels

        # Build EfficientNet blocks
        for expand_ratio, kernel_size, stride, out_channels, num_blocks in block_params:
            out_channels = int(out_channels * width_coeff)  # Scale output channels by width coefficient
            num_blocks = int(num_blocks * depth_coeff)      # Scale number of blocks by depth coefficient
            for i in range(num_blocks):
                stride = stride if i == 0 else 1  # Only apply stride to the first block in each stage
                self.blocks.append(MBConvBlock(in_channels, out_channels, expand_ratio, kernel_size, stride))
                in_channels = out_channels  # Update input channels for next block

        # Head of the network
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),             # Flatten to prepare for the classifier
            nn.Dropout(dropout_rate), # Dropout for regularization
            nn.Linear(final_channels, num_classes)  # Final linear layer for classification
        )

    def forward(self, x):
        x = self.stem(x)  # Pass through stem
        for block in self.blocks:
            x = block(x)  # Pass through each block
        x = self.head(x)  # Pass through the head
        return x

# Function to create EfficientNet-B0 model
def efficientnet_b0(pretrained=False):
    model = EfficientNet(width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['efficientnet_b0'])
        model.load_state_dict(state_dict)
    return model

# Function to create EfficientNet-B1 model
def efficientnet_b1(pretrained=False):
    model = EfficientNet(width_coeff=1.0, depth_coeff=1.1, dropout_rate=0.2)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['efficientnet_b1'])
        model.load_state_dict(state_dict)
    return model

# Function to calculate the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# A basic training loop for EfficientNet
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

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

        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# A basic evaluation loop for EfficientNet
def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

# Learning rate scheduler to adjust learning rate during training
def adjust_learning_rate(optimizer, epoch, initial_lr, decay_factor=0.1, decay_epochs=30):
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Model initialization and training function
def initialize_and_train(train_loader, val_loader, model_fn, learning_rate, num_epochs, device):
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, initial_lr=learning_rate)

        # Training
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_efficientnet_model.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    return model

# Loading the best model weights
def load_best_model(model_fn, device):
    model = model_fn().to(device)
    model.load_state_dict(torch.load('best_efficientnet_model.pth'))
    return model

# Function to test the model on test data
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc

# Utility function to predict a single image
def predict_single_image(model, image, device):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # Add batch dimension
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Data augmentation pipeline for training dataset
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to prepare the data loaders
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def prepare_data_loaders(data_dir, batch_size):
    train_dataset = ImageFolder(root=f'{data_dir}/train', transform=train_transforms)
    val_dataset = ImageFolder(root=f'{data_dir}/val', transform=val_transforms)
    test_dataset = ImageFolder(root=f'{data_dir}/test', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Custom dataset class for handling non-standard image data formats
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Custom collate function for handling batches with different image sizes
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack([torch.Tensor(img) for img in images], dim=0)
    labels = torch.tensor(labels)
    return images, labels

# Function to fine-tune EfficientNet on a new dataset
def fine_tune_model(model, train_loader, val_loader, num_classes, learning_rate, num_epochs, device):
    # Modify the final layer of the model to match the number of classes
    model.head[-1] = nn.Linear(model.head[-1].in_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, initial_lr=learning_rate)

        # Training
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'fine_tuned_efficientnet_model.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    return model

# Function to load the fine-tuned model
def load_fine_tuned_model(model_fn, num_classes, device):
    model = model_fn().to(device)
    model.head[-1] = nn.Linear(model.head[-1].in_features, num_classes).to(device)
    model.load_state_dict(torch.load('fine_tuned_efficientnet_model.pth'))
    return model

# Function to calculate model inference time
import time

def measure_inference_time(model, test_loader, device, num_trials=100):
    model.eval()  # Set model to evaluation mode
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_trials:
                break
            inputs = inputs.to(device)

            start_time = time.time()
            model(inputs)
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f'Average inference time per batch: {avg_time:.6f} seconds')
    return avg_time

# Function to calculate model accuracy and confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_accuracy_and_confusion_matrix(model, test_loader, num_classes, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    print(f'Accuracy: {acc:.4f}')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return acc, cm

# Function to freeze the layers of the model for transfer learning
def freeze_model_layers(model, freeze_upto_layer=0):
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < freeze_upto_layer:
            param.requires_grad = False

# Transfer learning: loading a pretrained EfficientNet and fine-tuning it
def perform_transfer_learning(train_loader, val_loader, num_classes, device):
    model = efficientnet_b0(pretrained=True).to(device)

    # Freeze all layers except the final layer
    freeze_model_layers(model, freeze_upto_layer=len(list(model.named_parameters())) - 1)

    # Fine-tune the model on a new dataset
    fine_tuned_model = fine_tune_model(model, train_loader, val_loader, num_classes, learning_rate=1e-4, num_epochs=10, device=device)
    return fine_tuned_model

# Grad-CAM implementation for model interpretability
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_extractor = create_feature_extractor(self.model, return_nodes={target_layer: 'activation'})
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        activation = self.feature_extractor(x)['activation']
        activation.register_hook(self.save_gradient)

        output = self.model(x)
        output[:, output.max(1)[-1]].backward()
        return activation

    def generate_heatmap(self, activation):
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        for i in range(activation.size(1)):
            activation[:, i, :, :] *= pooled_gradients[i]
        heatmap = activation.mean(dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()
        return heatmap

    def visualize_cam(self, image, heatmap, alpha=0.4):
        heatmap = np.uint8(255 * heatmap)
        heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
        heatmap = Image.fromarray(heatmap).resize(image.size)
        heatmap = np.array(heatmap)
        superimposed_img = heatmap * alpha + np.array(image)
        return Image.fromarray(np.uint8(superimposed_img))

# Function to apply Grad-CAM on a single image
def apply_grad_cam(model, image, target_layer, device):
    grad_cam = GradCAM(model, target_layer)
    model.eval()
    image = image.to(device).unsqueeze(0)

    # Get the activation and gradients
    activation = grad_cam(image)
    heatmap = grad_cam.generate_heatmap(activation)

    # Convert the image tensor to PIL for visualization
    image_pil = transforms.ToPILImage()(image.squeeze().cpu())

    # Visualize the heatmap
    return grad_cam.visualize_cam(image_pil, heatmap)