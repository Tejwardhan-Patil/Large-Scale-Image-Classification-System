import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

# Custom initialization of ResNet with weights
def initialize_resnet(model, pretrained_weights_path=None, custom_weights_init=False):
    if pretrained_weights_path:
        model.load_state_dict(torch.load(pretrained_weights_path))
    elif custom_weights_init:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model

# ResNet variations with different input configurations
class ResNetVariant(nn.Module):
    def __init__(self, base_model, input_channels=3, num_classes=1000):
        super(ResNetVariant, self).__init__()
        self.base_model = base_model
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * base_model.block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Function to create a custom ResNet variant
def custom_resnet_variant(input_channels=3, num_classes=1000, variant="resnet50", pretrained_weights_path=None):
    model_map = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    base_model = model_map[variant](num_classes=num_classes)
    model = ResNetVariant(base_model, input_channels=input_channels, num_classes=num_classes)
    
    if pretrained_weights_path:
        model = initialize_resnet(model, pretrained_weights_path)
    
    return model

# Block variants for deeper layers
class DeepBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, downsample=None, activation=F.relu):
        super(DeepBasicBlock, self).__init__(in_planes, planes, stride, downsample)
        self.activation = activation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class DeepBottleneck(Bottleneck):
    def __init__(self, in_planes, planes, stride=1, downsample=None, activation=F.relu):
        super(DeepBottleneck, self).__init__(in_planes, planes, stride, downsample)
        self.activation = activation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

# Custom ResNet with deep blocks
def deep_resnet50(num_classes=1000):
    return ResNet(DeepBottleneck, [3, 4, 6, 3], num_classes)

def deep_resnet101(num_classes=1000):
    return ResNet(DeepBottleneck, [3, 4, 23, 3], num_classes)

def deep_resnet152(num_classes=1000):
    return ResNet(DeepBottleneck, [3, 8, 36, 3], num_classes)

# Utility function for adjusting the learning rate during training
def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_rate=0.1, lr_decay_epochs=[30, 60, 90]):
    """Adjusts the learning rate based on the epoch and decay schedule."""
    lr = initial_lr
    for milestone in lr_decay_epochs:
        if epoch >= milestone:
            lr *= lr_decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Gradient clipping utility for better training stability
def clip_gradient(optimizer, max_norm):
    """Clips gradients during backpropagation to avoid exploding gradients."""
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups, max_norm)

# A utility to freeze model layers during fine-tuning
def freeze_model_layers(model, freeze_until_layer):
    """Freezes layers until a certain layer is reached, useful for fine-tuning."""
    for name, param in model.named_parameters():
        if name.startswith(freeze_until_layer):
            break
        param.requires_grad = False

# Custom learning rate scheduler
class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, warmup_epochs=5, warmup_start_lr=1e-6, target_lr=1e-2):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) *
                    (self.last_epoch / self.warmup_epochs) for _ in self.base_lrs]
        return [self.target_lr for _ in self.base_lrs]

# Training loop with gradient accumulation
def train_model(model, dataloader, criterion, optimizer, num_epochs=100, accumulation_steps=4, device='cuda'):
    """Trains the model using gradient accumulation for memory-efficient training."""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

# A testing utility to evaluate model performance
def evaluate_model(model, dataloader, criterion, device='cuda'):
    """Evaluates model performance on a validation or test dataset."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return total_loss, accuracy

# ResNet ensemble model class for ensembling multiple ResNet models
class ResNetEnsemble(nn.Module):
    def __init__(self, models, num_classes=1000):
        super(ResNetEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.fc = nn.Linear(len(models) * 512 * models[0].block.expansion, num_classes)

    def forward(self, x):
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)

        ensemble_output = torch.cat(outputs, dim=1)
        ensemble_output = self.fc(ensemble_output)
        return ensemble_output

# Function to create an ensemble of ResNet models
def create_resnet_ensemble(model_names, num_classes=1000, pretrained_weights_paths=None):
    models = []
    for i, model_name in enumerate(model_names):
        model_map = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152
        }
        model = model_map[model_name](num_classes=num_classes)
        if pretrained_weights_paths and pretrained_weights_paths[i]:
            model = initialize_resnet(model, pretrained_weights_paths[i])
        models.append(model)

    return ResNetEnsemble(models, num_classes=num_classes)

# Fine-tuning utility function for transfer learning
def fine_tune_resnet(model, dataloader, criterion, optimizer, fine_tune_layers=None, num_epochs=50, device='cuda'):
    """Fine-tunes the ResNet model on a new dataset."""
    if fine_tune_layers:
        freeze_model_layers(model, fine_tune_layers)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

# Gradient checkpointing for memory optimization
def checkpoint_resnet_forward(model, inputs):
    """Uses gradient checkpointing to save memory during forward pass."""
    from torch.utils.checkpoint import checkpoint
    return checkpoint(model, inputs)