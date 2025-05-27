# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import os,sys
cwd = os.getcwd()
sys.path.append(cwd)
import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import contextlib
import io
from PIL import Image

# input image, output compressibility score
def jpeg_compressibility(images):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    pil_images = images
    if isinstance(images, np.ndarray):
        pil_images = [Image.fromarray(image) for image in images]
        
    # buffers = [io.BytesIO() for _ in images]
    # for image, buffer in zip(images, buffers):
    #     image.save(buffer, format="JPEG", quality=95)
    # sizes = [buffer.tell() / 1000 for buffer in buffers]
    sizes = []
    with contextlib.ExitStack() as stack:
        buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / 1000)  # Size in kilobytes
    
    return -np.array(sizes)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SinusoidalTimeConvNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, time_encoding_dim=64, distributional=False, bin_num=256, bin_min= -255, bin_max=0, dtype=torch.float32):
        super(SinusoidalTimeConvNet, self).__init__()
        
        self.dtype = dtype
        self.time_encoding_dim = time_encoding_dim

        # bin tricks
        self.distributional = distributional
        self.bin_num = bin_num
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_edges = torch.linspace(self.bin_min, self.bin_max, self.bin_num + 1)  # shape: [num_bins + 1]
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0  # shape: [num_bins]
        # Standard convolutional layers
        self.layer1 = ResidualBlock(num_channels, 64, stride=1)
        self.layer2 = ResidualBlock(64 + time_encoding_dim, 128, stride=2)  # Concatenating time embedding here
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.distributional:
            self.fc = nn.Linear(256, num_classes * bin_num)
        else:
            self.fc = nn.Linear(256, num_classes)

    def sinusoidal_encoding(self, timesteps, height, width):
        # Normalize timesteps to be in the range [0, 1]
        timesteps = timesteps.float() / 1000.0  # Assuming timesteps are provided as integers

        # Generate a series of frequencies for the sinusoidal embeddings
        frequencies = torch.exp(torch.arange(0, self.time_encoding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.time_encoding_dim))
        frequencies = frequencies.to(timesteps.device)

        # Apply the frequencies to the timesteps
        arguments = timesteps[:, None] * frequencies[None, :]
        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)

        # Reshape the time embedding to match the spatial dimensions (height, width)
        encoding = encoding[:, :, None, None].repeat(1, 1, height, width)  # Repeat for spatial dimensions
        return encoding

    def forward(self, x, timesteps, eta = None):
        batch_size, channels, height, width = x.size()

        # Pass through the first convolutional layer
        out = self.layer1(x.to(self.dtype))

        # Generate sinusoidal embeddings for the timesteps and expand to match the feature map dimensions
        timestep_embed = self.sinusoidal_encoding(timesteps, out.size(2), out.size(3))

        # Concatenate the time embedding with the output of the first layer
        combined_input = torch.cat([out, timestep_embed], dim=1)

        # Continue with the remaining convolutional layers
        out = self.layer2(combined_input)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten the feature map
        out = self.fc(out)
        
        if self.distributional and eta is not None:
            bin_edges = torch.linspace(self.bin_min, self.bin_max, self.bin_num + 1)  # shape: [num_bins + 1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # shape: [num_bins]
            probs = torch.softmax(out, dim=1)  # shape: [batch_size, num_bins]
            out = torch.logsumexp(torch.log(probs) + bin_centers.to(probs.device)/(self.bin_max - self.bin_min) * eta, dim=1)  # shape: [batch_size]
        return out

