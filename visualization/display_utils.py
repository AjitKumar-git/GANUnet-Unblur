import torch
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

from models import load_checkpoint

def load_model(model_class, model_path, device="cpu"):
    """
    Loads a model's weights from a given file path.

    Args:
        model_class (nn.Module): The model class to instantiate.
        model_path (str): Path to the saved model's state dictionary.
        device (torch.device or str): Device to load the model to.

    Returns:
        nn.Module: The model with loaded weights.
    """
    model = model_class()
    return load_checkpoint(model, model_path, device=device)


def show_random_samples(dataset, num_samples=4, image_size=(128, 128)):
    """
    Display random samples from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        num_samples (int): The number of random samples to display.
        image_size (tuple): The size to which the images should be resized for display.
    """
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    for i, idx in enumerate(indices):
        blurred_image, sharp_image = dataset[idx]

        # Convert the tensor to a numpy array for display
        blurred_image = blurred_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        sharp_image = sharp_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

        blurred_image = np.clip(blurred_image * 0.5 + 0.5, 0, 1)  # Undo normalization
        sharp_image = np.clip(sharp_image * 0.5 + 0.5, 0, 1)  # Undo normalization

        # Resize if needed
        if image_size:
            blurred_image = Image.fromarray((blurred_image * 255).astype(np.uint8)).resize(image_size)
            sharp_image = Image.fromarray((sharp_image * 255).astype(np.uint8)).resize(image_size)

        axes[i].imshow(np.concatenate([blurred_image, sharp_image], axis=1))  # Show side by side
        axes[i].axis('off')
    
    plt.show()


def preprocess_image(image_path, image_size=(128, 128)):
    """
    Preprocess an image by loading, resizing, and normalizing.

    Args:
        image_path (str): Path to the image to process.
        image_size (tuple): Desired image size.

    Returns:
        torch.Tensor: The processed image as a tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)

    # Convert to tensor and normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image)

    return image


def show_generated_images(generator, dataloader, device, num_images=16, image_size=(128, 128)):
    """
    Display generated images from the generator model.

    Args:
        generator (nn.Module): The generator model.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        device (torch.device): Device to run the model on.
        num_images (int): The number of images to display.
        image_size (tuple): The size to which the images should be resized for display.
    """
    generator.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Get a batch of blurred images from the dataloader
        batch = next(iter(dataloader))
        blurred_images = batch[0].to(device)

        num_images = min(num_images, blurred_images.size(0))
        # Generate images
        generated_images = generator(blurred_images[:num_images])

        # Convert generated images to numpy for display
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)
        generated_images = generated_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Plot images
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            axes[i].imshow(generated_images[i])
            axes[i].axis('off')
        plt.show()


def save_generated_images(generator, blurred_images, epoch, device, image_size=(128, 128), save_dir="generated_images"):
    """
    Save generated images from the generator.

    Args:
        generator (nn.Module): The generator model.
        blurred_images (Tensor): A batch of blurred images.
        epoch (int): The current epoch number for saving purposes.
        device (torch.device): Device to run the model on.
        image_size (tuple): The size to which the images should be resized.
        save_dir (str): Directory to save generated images.
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()  # Set to evaluation mode
    with torch.no_grad():
        blurred_images = blurred_images.to(device)
        # Generate a batch of images
        generated_images = generator(blurred_images)

        # Un-normalize from [-1, 1] to [0, 1] and clamp
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)

        # torchvision.utils.make_grid expects (B, C, H, W) tensor
        grid = torchvision.utils.make_grid(generated_images.cpu(), nrow=4, normalize=False)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
        plt.axis("off")
        save_path = os.path.join(save_dir, f"generated_epoch_{epoch+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Generated images saved to {save_path}")