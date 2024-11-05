import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101

class CustomResNet(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrained=True, use_resnet101=False):
        super(CustomResNet, self).__init__()
        self.use_resnet101 = use_resnet101
        
        # Select between ResNet50 and ResNet101 as the backbone
        if self.use_resnet101:
            self.backbone = resnet101(pretrained=pretrained)
        else:
            self.backbone = resnet50(pretrained=pretrained)

        # Remove the default FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Custom fully connected layers
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)

        # Custom BatchNorm layers for additional regularization
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)

        # Initialize weights of custom layers
        self._initialize_weights()

    def forward(self, x):
        # Pass input through ResNet backbone
        x = self.backbone(x)

        # Custom fully connected layers with batch normalization and dropout
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        # Initialize weights of custom layers using Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        # Freeze all layers in the ResNet backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        # Unfreeze all layers in the ResNet backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

def initialize_model(num_classes, dropout_rate=0.5, pretrained=True, use_resnet101=False):
    """
    Initializes and returns the custom ResNet model.
    Can choose between ResNet50 and ResNet101 as the backbone.
    """
    model = CustomResNet(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained, use_resnet101=use_resnet101)
    return model

if __name__ == "__main__":
    # Initialize a custom ResNet with ResNet101 for 10 classes
    model = initialize_model(num_classes=10, dropout_rate=0.3, pretrained=True, use_resnet101=True)
    
    # Print model architecture to verify correct initialization
    print(model)

    # Input tensor with batch size 1 and 3-channel image (224x224)
    sample_input = torch.randn(1, 3, 224, 224)

    # Forward pass through the model
    output = model(sample_input)

    # Print output shape to confirm model works as expected
    print(f"Output shape: {output.shape}")

    # Function to count the total number of parameters in the model
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Function to print detailed layer-wise summary of the model
    def print_summary(self, input_size=(3, 224, 224)):
        from torchsummary import summary
        summary(self, input_size)

    # Method to partially freeze layers of the ResNet backbone
    def freeze_partial_backbone(self, freeze_up_to_layer=5):
        """
        Freezes a portion of the ResNet backbone layers up to a specified layer.
        freeze_up_to_layer: Layer number to freeze up to. This allows fine-tuning
                            only a subset of the layers.
        """
        layer_count = 0
        for child in self.backbone.children():
            layer_count += 1
            if layer_count <= freeze_up_to_layer:
                for param in child.parameters():
                    param.requires_grad = False

    # Method to selectively unfreeze specific layers of the model
    def unfreeze_specific_layers(self, layers_to_unfreeze):
        """
        Unfreezes specific layers provided in the list `layers_to_unfreeze`.
        layers_to_unfreeze: List of layers that should be unfrozen.
        Example: ['layer1', 'layer2'] to unfreeze specific ResNet backbone layers.
        """
        for name, child in self.backbone.named_children():
            if name in layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True

    # Method to adjust the learning rate for different layers
    def adjust_learning_rate(self, base_lr, layerwise_lr_decay=0.9):
        """
        Adjusts the learning rate for different layers based on their depth.
        base_lr: The base learning rate.
        layerwise_lr_decay: A factor by which the learning rate decreases for deeper layers.
        """
        optimizer_params = []
        lr = base_lr
        for child in self.backbone.children():
            optimizer_params.append({'params': child.parameters(), 'lr': lr})
            lr *= layerwise_lr_decay
        return optimizer_params

    # Method to add custom layers on top of the ResNet backbone dynamically
    def add_custom_layers(self, layer_configs):
        """
        Dynamically adds custom layers on top of the ResNet backbone.
        layer_configs: A list of dictionaries where each dictionary defines a layer configuration.
                       Example: [{'type': 'fc', 'in_features': 512, 'out_features': 256}, {'type': 'dropout', 'rate': 0.5}]
        """
        layers = []
        for config in layer_configs:
            if config['type'] == 'fc':
                layers.append(nn.Linear(config['in_features'], config['out_features']))
                layers.append(nn.ReLU())
            elif config['type'] == 'dropout':
                layers.append(nn.Dropout(config['rate']))
            elif config['type'] == 'batchnorm':
                layers.append(nn.BatchNorm1d(config['num_features']))
        self.custom_layers = nn.Sequential(*layers)

    # Method to forward propagate with custom layers dynamically added
    def forward_with_custom_layers(self, x):
        """
        Forward propagation with dynamically added custom layers.
        """
        x = self.backbone(x)
        if hasattr(self, 'custom_layers'):
            x = self.custom_layers(x)
        return x

    # Function to load pre-trained weights from a specified file
    def load_pretrained_weights(self, filepath):
        """
        Loads pretrained weights from a given file.
        filepath: Path to the file containing the weights.
        """
        self.load_state_dict(torch.load(filepath))
        print(f"Pretrained weights loaded from {filepath}")

    # Function to save the model's weights to a specified file
    def save_model_weights(self, filepath):
        """
        Saves the current model weights to a file.
        filepath: Path where the weights will be saved.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

    # Function to load only the backbone weights (without affecting custom layers)
    def load_backbone_weights(self, filepath):
        """
        Loads pretrained weights for the backbone only.
        This is useful when custom layers are different from the original model.
        filepath: Path to the file containing backbone weights.
        """
        state_dict = torch.load(filepath)
        backbone_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone')}
        self.backbone.load_state_dict(backbone_dict)
        print(f"Backbone weights loaded from {filepath}")

    # Method to compute output embeddings instead of classification
    def compute_embeddings(self, x):
        """
        Instead of returning class predictions, this method computes the feature embeddings
        from the backbone and custom layers (without the final classification layer).
        """
        x = self.backbone(x)
        if hasattr(self, 'custom_layers'):
            x = self.custom_layers(x)
        return x

    # Method to fine-tune the model on a new dataset with different input size
    def fine_tune(self, dataset, input_size=(3, 128, 128), num_epochs=10, learning_rate=1e-3):
        """
        Fine-tunes the model on a new dataset.
        dataset: The dataset for fine-tuning.
        input_size: Input size of images in the dataset.
        num_epochs: Number of epochs for fine-tuning.
        learning_rate: Learning rate for fine-tuning.
        """
        from torch.utils.data import DataLoader
        from torch.optim import Adam

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs = inputs.view(-1, *input_size)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

        print("Fine-tuning complete.")

    # Method to implement Grad-CAM for visualizing class activation maps
    def grad_cam(self, x, target_class=None):
        """
        Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing
        which parts of the input image are most important for a specific class prediction.
        target_class: The class for which the activation map is generated.
        """
        from torch.autograd import Function
        import numpy as np
        import cv2

        class GradCamHook(Function):
            def __init__(self, model, target_class):
                super().__init__()
                self.model = model
                self.target_class = target_class
                self.gradients = None

            def hook_layers(self):
                def forward_hook(module, input, output):
                    self.activation_output = output

                def backward_hook(module, grad_input, grad_output):
                    self.gradients = grad_output[0]

                self.model.backbone.layer4[2].register_forward_hook(forward_hook)
                self.model.backbone.layer4[2].register_backward_hook(backward_hook)

        # Initialize hook
        gradcam = GradCamHook(self, target_class)
        gradcam.hook_layers()

        # Forward pass
        output = self.forward(x)
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())

        # Backward pass
        self.zero_grad()
        output[:, target_class].backward(retain_graph=True)

        # Get gradients and activation maps
        gradients = gradcam.gradients.cpu().data.numpy()[0]
        activation_output = gradcam.activation_output.cpu().data.numpy()[0]

        # Pool gradients across channels
        pooled_gradients = np.mean(gradients, axis=(1, 2))

        # Weight the activations by the pooled gradients
        for i in range(activation_output.shape[0]):
            activation_output[i, :, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = np.mean(activation_output, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[3]))

        return heatmap

    # Function to overlay heatmap on the input image
    def overlay_heatmap(self, heatmap, img, intensity=0.5):
        """
        Overlays the heatmap generated by Grad-CAM onto the input image.
        img: The input image (original image).
        intensity: The intensity of the heatmap overlay.
        """
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap * intensity + np.float32(img) / 255
        overlay = overlay / np.max(overlay)
        plt.imshow(overlay)
        plt.show()

    # Method to integrate focal loss in the model
    def focal_loss(self, outputs, targets, alpha=1, gamma=2):
        """
        Implements focal loss for dealing with class imbalance in classification.
        outputs: The predicted outputs from the model.
        targets: The true labels.
        alpha: Weighting factor for balancing the classes.
        gamma: Focusing parameter to down-weight easy examples.
        """
        bce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    # Method to add additional custom attention mechanism to the model
    def add_attention_layer(self, attention_type='self'):
        """
        Adds an attention layer to the model.
        attention_type: Type of attention ('self', 'spatial', 'channel').
        """
        if attention_type == 'self':
            self.attention_layer = SelfAttention(self.backbone.fc.in_features)
        elif attention_type == 'spatial':
            self.attention_layer = SpatialAttention()
        elif attention_type == 'channel':
            self.attention_layer = ChannelAttention(self.backbone.fc.in_features)
        else:
            raise ValueError("Unsupported attention type")

    def forward_with_attention(self, x):
        """
        Forward propagation with attention mechanism applied.
        """
        x = self.backbone(x)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x) * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x