import torch
import torch.nn as nn
from particle_detection.vae.model import create_vae

def load_vae(model_path, device="cuda", latent_dim=128):
    vae = create_vae(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(model_path))
    vae = nn.DataParallel(vae).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE model loaded successfully")
    return vae