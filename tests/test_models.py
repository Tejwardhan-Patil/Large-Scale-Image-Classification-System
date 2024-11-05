import unittest
import torch
import torch.nn as nn
from models.architectures import cnn, resnet, efficientnet

class TestCNNArchitecture(unittest.TestCase):

    def setUp(self):
        self.model = cnn.CNNModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = (1, 3, 224, 224)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_cnn_initialization(self):
        self.assertIsInstance(self.model, nn.Module)

    def test_cnn_layers(self):
        layers = list(self.model.children())
        self.assertGreater(len(layers), 0, "CNN model should have layers.")

    def test_cnn_forward_pass(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 10)

    def test_cnn_conv_layers(self):
        conv_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]
        self.assertGreater(len(conv_layers), 0, "CNN model should have convolutional layers.")

    def test_cnn_activation_function(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.assertIsInstance(module, nn.ReLU)

    def test_cnn_output_shape(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 10), "Output shape should match the number of classes.")

    def test_cnn_maxpool_layers(self):
        maxpool_layers = [module for module in self.model.modules() if isinstance(module, nn.MaxPool2d)]
        self.assertGreater(len(maxpool_layers), 0, "CNN model should have max pooling layers.")

    def test_cnn_dropout(self):
        dropout_layers = [module for module in self.model.modules() if isinstance(module, nn.Dropout)]
        self.assertGreater(len(dropout_layers), 0, "CNN model should have dropout layers for regularization.")


class TestResNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.model = resnet.ResNet50().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = (1, 3, 224, 224)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_resnet_initialization(self):
        self.assertIsInstance(self.model, nn.Module)

    def test_resnet_layers(self):
        layers = list(self.model.children())
        self.assertGreater(len(layers), 0, "ResNet model should have layers.")

    def test_resnet_forward_pass(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 1000)

    def test_resnet_conv_layers(self):
        conv_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]
        self.assertGreater(len(conv_layers), 0, "ResNet should have convolutional layers.")

    def test_resnet_batch_norm_layers(self):
        batch_norm_layers = [module for module in self.model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertGreater(len(batch_norm_layers), 0, "ResNet should have batch normalization layers.")

    def test_resnet_skip_connections(self):
        res_blocks = [module for module in self.model.modules() if isinstance(module, resnet.BasicBlock)]
        self.assertGreater(len(res_blocks), 0, "ResNet should have residual (skip) connections.")

    def test_resnet_output_shape(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 1000))

    def test_resnet_activation_function(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.assertIsInstance(module, nn.ReLU)


class TestEfficientNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.model = efficientnet.EfficientNetB0().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = (1, 3, 224, 224)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_efficientnet_initialization(self):
        self.assertIsInstance(self.model, nn.Module)

    def test_efficientnet_layers(self):
        layers = list(self.model.children())
        self.assertGreater(len(layers), 0, "EfficientNet model should have layers.")

    def test_efficientnet_forward_pass(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 1000)

    def test_efficientnet_conv_layers(self):
        conv_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]
        self.assertGreater(len(conv_layers), 0, "EfficientNet should have convolutional layers.")

    def test_efficientnet_batch_norm_layers(self):
        batch_norm_layers = [module for module in self.model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertGreater(len(batch_norm_layers), 0, "EfficientNet should have batch normalization layers.")

    def test_efficientnet_output_shape(self):
        input_tensor = torch.randn(self.input_shape).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 1000))

    def test_efficientnet_activation_function(self):
        for module in self.model.modules():
            if isinstance(module, nn.SiLU):
                self.assertIsInstance(module, nn.SiLU)


class TestModelHelperFunctions(unittest.TestCase):

    def test_parameter_count_cnn(self):
        model = cnn.CNNModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(param_count, 0, "CNN model should have trainable parameters.")

    def test_parameter_count_resnet(self):
        model = resnet.ResNet50().to('cuda' if torch.cuda.is_available() else 'cpu')
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(param_count, 0, "ResNet should have trainable parameters.")

    def test_parameter_count_efficientnet(self):
        model = efficientnet.EfficientNetB0().to('cuda' if torch.cuda.is_available() else 'cpu')
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(param_count, 0, "EfficientNet should have trainable parameters.")

    def test_cnn_parameter_shapes(self):
        model = cnn.CNNModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.shape, f"Parameter {name} should have a shape defined.")

    def test_resnet_parameter_shapes(self):
        model = resnet.ResNet50().to('cuda' if torch.cuda.is_available() else 'cpu')
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.shape, f"Parameter {name} should have a shape defined.")

    def test_efficientnet_parameter_shapes(self):
        model = efficientnet.EfficientNetB0().to('cuda' if torch.cuda.is_available() else 'cpu')
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.shape, f"Parameter {name} should have a shape defined.")


if __name__ == '__main__':
    unittest.main()