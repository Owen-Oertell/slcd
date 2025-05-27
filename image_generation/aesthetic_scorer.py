from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel
import math
from torch.utils.checkpoint import checkpoint


ASSETS_PATH = resources.files("assets")

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1), embed
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed



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
        self.layer_m1 = ResidualBlock(128, 128, stride=1)
        self.layer_m2 = ResidualBlock(128, 128, stride=1)
        self.layer_m3 = ResidualBlock(128, 128, stride=1)
        self.layer_m4 = ResidualBlock(128, 128, stride=1)
        self.layer_m5 = ResidualBlock(128, 128, stride=1)
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
        out = self.layer_m1(out)
        out = self.layer_m2(out)
        out = self.layer_m3(out)
        out = self.layer_m4(out)
        out = self.layer_m5(out)
        
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
