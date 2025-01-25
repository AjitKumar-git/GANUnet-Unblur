# load_one_folder.py
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImagePairHandler(Dataset):
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # List all image files in the directory
        self.image_files = sorted(os.listdir(root_dir))

        self.image_pairs = []

        # Pair images by filename (handle blur and sharp suffix/prefix cases)
        for image_file in self.image_files:
            if not any(suffix in image_file.lower() for suffix in ['blur', 'blurred', 'sharp', 'sharpened', 'gt']):
                # Image with no suffix/prefix: find corresponding blurred and sharp images
                blurred_image = image_file
                sharp_image = image_file.replace('blur', 'sharp').replace('blurred', 'sharpened')
                
                if any(blurred_image.lower() in img.lower() for img in self.image_files if 'blur' in img.lower()):
                    self.image_pairs.append((blurred_image, sharp_image))

            elif 'blur' in image_file.lower() or 'blurred' in image_file.lower():
                # Handle blurred images
                blurred_image = image_file
                sharp_image = image_file.replace('blur', 'sharp').replace('blurred', 'sharpened')
                
                if sharp_image in self.image_files:
                    self.image_pairs.append((blurred_image, sharp_image))

            elif 'sharp' in image_file.lower() or 'sharpened' in image_file.lower() or 'gt' in image_file.lower():
                # Handle sharp or ground truth images
                sharp_image = image_file
                blurred_image = image_file.replace('sharp', 'blur').replace('sharpened', 'blurred')
                
                if blurred_image in self.image_files:
                    self.image_pairs.append((blurred_image, sharp_image))

        random.shuffle(self.image_pairs)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        blurred_filename, sharp_filename = self.image_pairs[index]
        blurred_img = Image.open(os.path.join(self.root_dir, blurred_filename)).convert('RGB')
        sharp_img = Image.open(os.path.join(self.root_dir, sharp_filename)).convert('RGB')
        blurred_img = self.transform(blurred_img)
        sharp_img = self.transform(sharp_img)
        return blurred_img, sharp_img
