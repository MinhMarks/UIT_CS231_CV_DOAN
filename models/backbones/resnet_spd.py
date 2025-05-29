import torch
import torch.nn as nn
import torchvision

class space_to_depth(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if stride == 2:
            layers2 = [
                nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
                space_to_depth(),
                nn.BatchNorm2d(4 * out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            ]
        else:
            layers2 = [
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            ]

        layers.extend(layers2)
        self.residual_function = nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, layers_to_freeze=2, layers_to_crop=[]):
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        # Initialize block and num_block based on model_name
        if 'resnet50' in model_name:
            num_block = [3, 4, 6, 3]
        elif 'resnet101' in model_name:
            num_block = [3, 4, 23, 3]
        elif 'resnet152' in model_name:
            num_block = [3, 8, 36, 3]
        elif 'resnet34' in model_name or 'resnet18' in model_name:
            num_block = [3, 4, 6, 3]  # Adjust for smaller models if needed
        else:
            raise NotImplementedError('Backbone architecture not recognized!')

        self.in_channels = 64
        self.conv1 = Focus(3, 64, k=1, s=1)

        self.conv2_x = self._make_layer(BottleNeck, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(BottleNeck, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(BottleNeck, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(BottleNeck, 512, num_block[3], 2)

        # Remove avg_pool and fc
        self.avg_pool = None
        self.fc = None

        # Handle pretrained weights
        if pretrained:
            # Note: This implementation does not support pretrained weights natively.
            # You can add torchvision.models.resnet50(weights='IMAGENET1K_V1') and transfer weights manually if needed.
            pass

        # Freeze layers
        if pretrained and layers_to_freeze >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        if pretrained and layers_to_freeze >= 1:
            for param in self.conv2_x.parameters():
                param.requires_grad = False
        if pretrained and layers_to_freeze >= 2:
            for param in self.conv3_x.parameters():
                param.requires_grad = False
        if pretrained and layers_to_freeze >= 3:
            for param in self.conv4_x.parameters():
                param.requires_grad = False

        # Crop layers
        if 4 in layers_to_crop:
            self.conv5_x = None
        if 3 in layers_to_crop:
            self.conv4_x = None

        # Calculate out_channels
        out_channels = 2048
        if 'resnet34' in model_name or 'resnet18' in model_name:
            out_channels = 512
        self.out_channels = out_channels // 2 if self.conv5_x is None else out_channels
        self.out_channels = self.out_channels // 2 if self.conv4_x is None else self.out_channels

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        if self.conv4_x is not None:
            x = self.conv4_x(x)
        if self.conv5_x is not None:
            x = self.conv5_x(x)
        return x