import torch
import torch.nn as nn

class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Grayscale input, 64 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: (H, W) -> (H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: (H/2, W/2) -> (H/4, W/4)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: (H/4, W/4) -> (H/8, W/8)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 512 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: (H/8, W/8) -> (H/16, W/16)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output 1 channel for grayscale
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_autoencoder():
  encoder = Encoder()
  return Autoencoder(encoder)

