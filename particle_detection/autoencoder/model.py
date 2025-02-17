import torch
from torch import nn
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
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

def create_autoencoder():
    #Creates an Autoencoder with a pretrained VGG16 encoder, modified for grayscale images.
    pretrained_encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    #pretrained_encoder = models.vgg16(weights.IMAGENET1K_V1)

    # for 1024x1024
    '''
    new_first_layer = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )
    '''
    new_first_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # Copy weights (average RGB channels to single grayscale channel)
    new_first_layer.weight.data = pretrained_encoder.features[0].weight.mean(dim=1, keepdim=True).detach()

    # Replace the first layer in the encoder
    pretrained_encoder.features[0] = new_first_layer

    # Remove last layer (classifier part)
    encoder = nn.Sequential(*list(pretrained_encoder.features.children())[:-1])
    #for 512x4x4 (smaller latent space)
    #encoder = nn.Sequential(*list(pretrained_encoder.features.children()))

    return Autoencoder(encoder)

"""
def create_autoencoder():
    Creates an Autoencoder with a pretrained VGG16 encoder.
    pretrained_encoder = models.vgg16(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(pretrained_encoder.features.children())[:-1])
    return Autoencoder(encoder)
"""
