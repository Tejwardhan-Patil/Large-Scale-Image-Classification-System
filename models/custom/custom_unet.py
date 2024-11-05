import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        output = self.conv(dec1)
        return torch.sigmoid(output)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(AttentionUNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.attention3 = AttentionBlock(F_g=init_features * 8, F_l=init_features * 8, F_int=init_features * 4)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.attention2 = AttentionBlock(F_g=init_features * 4, F_l=init_features * 4, F_int=init_features * 2)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)
        self.attention1 = AttentionBlock(F_g=init_features * 2, F_l=init_features * 2, F_int=init_features)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, self.attention3(enc4, enc4))
        dec2 = self.up2(dec3, self.attention2(enc3, enc3))
        dec1 = self.up1(dec2, self.attention1(enc2, enc2))

        output = self.conv(dec1)
        return torch.sigmoid(output)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UNetWithResiduals(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetWithResiduals, self).__init__()

        self.encoder1 = ResidualBlock(in_channels)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        output = self.conv(dec1)
        return torch.sigmoid(output)

class DeepSupervisionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepSupervisionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

class UNetWithDeepSupervision(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetWithDeepSupervision, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.final_conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

        self.supervision1 = DeepSupervisionBlock(init_features * 8, out_channels)
        self.supervision2 = DeepSupervisionBlock(init_features * 4, out_channels)
        self.supervision3 = DeepSupervisionBlock(init_features * 2, out_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        output = self.final_conv(dec1)
        
        supervision3 = self.supervision3(dec1)
        supervision2 = self.supervision2(dec2)
        supervision1 = self.supervision1(dec3)

        return torch.sigmoid(output), torch.sigmoid(supervision1), torch.sigmoid(supervision2), torch.sigmoid(supervision3)

class MultiScaleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(MultiScaleUNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

        self.scale1 = nn.Conv2d(init_features * 8, out_channels, kernel_size=1)
        self.scale2 = nn.Conv2d(init_features * 4, out_channels, kernel_size=1)
        self.scale3 = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        output = self.conv(dec1)
        scale1_output = self.scale1(enc4)
        scale2_output = self.scale2(enc3)
        scale3_output = self.scale3(enc2)

        return torch.sigmoid(output), torch.sigmoid(scale1_output), torch.sigmoid(scale2_output), torch.sigmoid(scale3_output)

class UNetWithSkipConnections(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetWithSkipConnections, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        skip_connections = torch.cat((dec1, enc1), dim=1)
        output = self.conv(skip_connections)
        return torch.sigmoid(output)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class UNetWithDenseBlocks(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, growth_rate=16, num_layers=4):
        super(UNetWithDenseBlocks, self).__init__()

        self.encoder1 = DenseBlock(in_channels, growth_rate, num_layers)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.encoder2 = DenseBlock(init_features * 2, growth_rate, num_layers)
        self.down2 = DownConvBlock(init_features * 2 + growth_rate * num_layers, init_features * 4)

        self.bottleneck = Bottleneck(init_features * 4, init_features * 8)

        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.decoder2 = DenseBlock(init_features * 4 + growth_rate * num_layers, growth_rate, num_layers)
        self.up1 = UpConvBlock(init_features * 4 + growth_rate * num_layers, init_features * 2)
        self.decoder1 = DenseBlock(init_features * 2 + growth_rate * num_layers, growth_rate, num_layers)

        self.conv = nn.Conv2d(init_features * 2 + growth_rate * num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.encoder2(enc2)
        enc4 = self.down2(enc3)

        bottleneck = self.bottleneck(enc4)

        dec2 = self.up2(bottleneck, enc3)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2, enc2)
        dec1 = self.decoder1(dec1)

        output = self.conv(dec1)
        return torch.sigmoid(output)

class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, task_channels=[1, 1], init_features=32):
        super(MultiTaskUNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.task1_conv = nn.Conv2d(init_features * 2, task_channels[0], kernel_size=1)
        self.task2_conv = nn.Conv2d(init_features * 2, task_channels[1], kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        task1_output = self.task1_conv(dec1)
        task2_output = self.task2_conv(dec1)

        return torch.sigmoid(task1_output), torch.sigmoid(task2_output)

class UNetWithSqueezeExcitation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, se_ratio=16):
        super(UNetWithSqueezeExcitation, self).__init__()

        self.encoder1 = ConvBlock(in_channels, init_features)
        self.down1 = DownConvBlock(init_features, init_features * 2)
        self.down2 = DownConvBlock(init_features * 2, init_features * 4)
        self.down3 = DownConvBlock(init_features * 4, init_features * 8)

        self.bottleneck = Bottleneck(init_features * 8, init_features * 16)

        self.up3 = UpConvBlock(init_features * 16, init_features * 8)
        self.up2 = UpConvBlock(init_features * 8, init_features * 4)
        self.up1 = UpConvBlock(init_features * 4, init_features * 2)

        self.conv = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(init_features * 16, init_features * 16 // se_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(init_features * 16 // se_ratio, init_features * 16, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        bottleneck = self.bottleneck(enc4)

        # Squeeze and Excitation
        se = self.global_avg_pool(bottleneck)
        se = self.fc1(se)
        se = F.relu(se, inplace=True)
        se = self.fc2(se)
        se = self.sigmoid(se)
        bottleneck = bottleneck * se

        dec3 = self.up3(bottleneck, enc4)
        dec2 = self.up2(dec3, enc3)
        dec1 = self.up1(dec2, enc2)

        output = self.conv(dec1)
        return torch.sigmoid(output)