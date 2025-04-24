import torch
import torch.nn.functional as F

def extract_patches(image, patch_size=16, stride=8):
    """Extracts overlapping patches from an input image tensor.

    Uses `torch.unfold` to create sliding windows (patches) across the input image.
    This function assumes the input image tensor has the shape (N, C, H, W).

    Args:
        image (torch.Tensor): Input image tensor of shape (N, C, H, W).
        patch_size (int, optional): The height and width of each patch. Defaults to 16.
        stride (int, optional): The stride between patches. Defaults to 8.

    Returns:
        torch.Tensor: A tensor of extracted patches with shape (num_patches, C, patch_size, patch_size).
    """
    _, channels, height, width = image.shape
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, channels, patch_size, patch_size)
    return patches

def extract_latent_representations(data_loader, vae, patch_size=16, device="cuda", batch_size=512):
    """Extracts latent representations from images using a Variational Autoencoder (VAE).

    This function processes images from a dataloader, extracts patches from each image,
    resizes the patches to a standard input size, and passes the patches through the encoder
    of the provided VAE model to obtain latent representations.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of input images.
        vae (torch.nn.Module): The VAE model used for encoding patches. Must have an `encode` method.
        patch_size (int, optional): Size of each patch extracted from the input images. Defaults to 16.
        device (str, optional): The device to perform computations on ("cuda" or "cpu"). Defaults to "cuda".
        batch_size (int, optional): The batch size for processing patches through the encoder. Defaults to 512.

    Returns:
        list of torch.Tensor: A list where each element is a tensor containing the latent representations
        for one input image from the dataset.
    """
    vae.eval()
    all_latent_representations = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            for img in batch:
                img = img.unsqueeze(0)  # Add batch dimension
                patches = extract_patches(img, patch_size).to(device)
                patches = F.interpolate(patches, size=(128, 128), mode="bilinear", align_corners=False)

                latent_representations = torch.cat([
                    vae.module.encode(patches[i:i + batch_size])
                    for i in range(0, patches.shape[0], batch_size)
                ], dim=0)
                
                all_latent_representations.append(latent_representations)

    return all_latent_representations
