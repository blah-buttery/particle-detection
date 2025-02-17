import torch
from torch import nn
from torchvision import models
from particle_detection.autoencoder.model import create_autoencoder

class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=256, input_channels=1):
        super(VAE, self).__init__()

        self.encoder = encoder  # transferred VGG16 encoder

        # Latent space layers
        self.flattened_dim = 512 * 8 * 8 
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder 64×64 → 1024×1024
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder = decoder  # transferred decoder

    def modify_encoder_for_grayscale(self, encoder):
        """Modifies the first convolutional layer of the encoder to accept grayscale images."""
        first_layer = encoder[0]  # First Conv2D layer in VGG16
        new_first_layer = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding
        )

        # Copy existing weights by averaging across RGB channels
        new_first_layer.weight.data = first_layer.weight.mean(dim=1, keepdim=True).detach()
        new_first_layer.bias.data = first_layer.bias.data.clone().detach()

        # Replace the first layer in the encoder
        encoder[0] = new_first_layer

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)  # Feature extraction
        x = x.view(x.size(0), -1)  # Flatten

        # Latent space mapping
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)  # Stabilize KL loss
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.fc_decoder(z)
        x = x.view(-1, 512, 8, 8)  # Reshape before upsampling
        x = self.decoder(x)
        return x, mu, logvar

    def encode(self, x):
        """Extracts the latent representation (mu)"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        return mu  # Only return mean (mu) for clustering tasks

def create_vae(autoencoder_path, input_channels=1):
    """
    Creates a VAE with a pretrained VGG16 encoder and transfers AE weights.
    Supports both grayscale (1-channel) and RGB (3-channel) images.
    """
    # Load Pretrained Autoencoder
    autoencoder = create_autoencoder()
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    # Use the autoencoder's encoder and decoder
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    # Create VAE with transferred weights
    #vae = VAE(encoder, decoder, input_channels=1)
    vae = VAE(encoder, decoder, input_channels=input_channels)
    if input_channels ==1:
      vae.modify_encoder_for_grayscale(vae.encoder)

    return vae

