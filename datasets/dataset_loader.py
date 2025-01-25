import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.load_one_folder import ImagePairHandler
from datasets.load_two_folder import TwoFolderDataset
from datasets.real_blur_dataset import RealBlurDataset

def load_dataset(dataset_type, root_dir, blurred_folder=None, sharp_folder=None, gt_folder=None, image_size=(128, 128)):
    """
    Load the dataset based on the specified type.
    
    Args:
        dataset_type (str): Type of the dataset ('one_folder', 'two_folder', 'real_blur').
        root_dir (str): Root directory where dataset folders are located.
        blurred_folder (str, optional): Folder containing blurred images (for two_folder and real_blur).
        sharp_folder (str, optional): Folder containing sharp images (for two_folder).
        gt_folder (str, optional): Folder containing ground truth images (for real_blur).
        image_size (tuple): Target size for images.

    Returns:
        dataset: The loaded dataset.
    """
    if dataset_type == 'one_folder':
        if blurred_folder is not None:
            raise ValueError("For 'one_folder' dataset type, only root_dir should be provided.")
        return ImagePairHandler(root_dir=root_dir, image_size=image_size)
    
    elif dataset_type == 'two_folder':
        if blurred_folder is None or sharp_folder is None:
            raise ValueError("For 'two_folder' dataset type, both blurred_folder and sharp_folder must be provided.")
        return TwoFolderDataset(blurred_folder=blurred_folder, sharp_folder=sharp_folder, image_size=image_size)
    
    elif dataset_type == 'real_blur':
        if blurred_folder is not None or gt_folder is not None:
            raise ValueError("For 'real_blur' dataset type, both blurred_folder and gt_folder must be None.")
        return RealBlurDataset(root_dir=root_dir, image_size=image_size)
    
    else:
        raise ValueError(f"Unknown dataset type '{dataset_type}'")

def visualize_dataset(dataset, num_samples=5):
    """
    Visualize a few samples from the dataset.

    Args:
        dataset: The dataset to visualize.
        num_samples (int): Number of samples to visualize.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        blurred_img, sharp_img = dataset[i]

        # Convert tensor to numpy
        blurred_img = blurred_img.permute(1, 2, 0).numpy()
        sharp_img = sharp_img.permute(1, 2, 0).numpy()

        # Normalize the images back to [0, 1]
        blurred_img = (blurred_img - blurred_img.min()) / (blurred_img.max() - blurred_img.min())
        sharp_img = (sharp_img - sharp_img.min()) / (sharp_img.max() - sharp_img.min())

        # Plot the images
        axes[i].imshow(blurred_img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
        
    plt.show()