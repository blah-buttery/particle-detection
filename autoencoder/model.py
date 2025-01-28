import torch
import torch.nn as nn
from torchvision import models

class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        # Flatten latent space for clustering
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def build_autoencoder(pretrained=True, device=torch.device("cpu")):
    pretrained_encoder = models.vgg16(weights="IMAGENET1K_V1") #load vgg16 as encoder
    encoder = nn.Sequential(*list(pretrained_encoder.features.children())[:-1])
    model = Autoencoder(encoder)
    return model.to(device)

