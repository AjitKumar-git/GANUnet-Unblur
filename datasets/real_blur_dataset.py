# real_blur_dataset.py
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RealBlurDataset(Dataset):
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Get all scene directories
        self.scene_dirs = [os.path.join(root_dir, scene) for scene in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, scene))]

        # Collect paths for blurred and ground truth images
        self.image_pairs = []
        for scene_dir in self.scene_dirs:
            blurred_dir = os.path.join(scene_dir, 'blur')  # Folder with blurred images
            gt_dir = os.path.join(scene_dir, 'gt')  # Folder with ground truth images
            
            if os.path.exists(blurred_dir) and os.path.exists(gt_dir):
                blurred_images = sorted([os.path.join(blurred_dir, img) for img in os.listdir(blurred_dir)])
                gt_images = sorted([os.path.join(gt_dir, img) for img in os.listdir(gt_dir)])
                
                # Ensure that the blurred and gt images are paired correctly by name
                for b, g in zip(blurred_images, gt_images):
                    self.image_pairs.append((b, g))

        # Shuffle the dataset
        random.shuffle(self.image_pairs)

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        

    def __len__(self):
        # Return the total number of image pairs
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Get the image pair (blurred, ground truth)
        blurred_path, gt_path = self.image_pairs[index]

        # Load images
        blurred_img = Image.open(blurred_path).convert('RGB')
        sharp_img = Image.open(gt_path).convert('RGB')

        # Apply transformations
        blurred_img = self.transform(blurred_img)
        sharp_img = self.transform(sharp_img)

        return blurred_img, sharp_img