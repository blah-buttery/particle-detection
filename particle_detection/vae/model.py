import torch
from torch import nn
from torchvision import models

class VAE(nn.Module):
    def __init__(self, latent_dim=256, input_channels=1):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim

        # channels 32 -> 64 -> 128 -> 256 -> 512
        # image size 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # dynamically compute feature map size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 128, 128)
            encoded_output = self.encoder(dummy_input)
            self.flattened_dim = encoded_output.numel()
            self.encoded_shape = encoded_output.shape

        # latent space mapping
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # channels 256 -> 128 -> 64 -> 32 -> 1
        # image size 8 -> 16 -> 32 -> 64 -> 128
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode -> reparameterize -> decode
        # assert x.shape[2:] == (4, 4), f"Unexpected encoder output shape: {x.shape}"
        x = self.encoder(x).view(x.size(0), -1)

        # latent space
        mu = self.fc_mu(x)
        logvar = 10 * torch.tanh(self.fc_logvar(x))
        z = self.reparameterize(mu, logvar)

        # decode
        x = self.fc_decoder(z).view(-1, *self.encoded_shape[1:])  # Reshape back to feature map size
        x = self.decoder(x)
        return x, mu, logvar

    def encode(self, x):
        # get latent space mu
        # assert x.shape[2:] == (4, 4), f"Unexpected encoder output shape: {x.shape}"
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(x)
      
    def encode_features(self, x):
        return self.encoder(x)

def create_vae(latent_dim=256, input_channels=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(latent_dim=latent_dim, input_channels=input_channels).to(device)
    return vae