# load_two_folder.py
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TwoFolderDataset(Dataset):
    def __init__(self, blurred_folder, sharp_folder, image_size=(128, 128)):
        self.blurred_folder = blurred_folder
        self.sharp_folder = sharp_folder
        self.image_size = image_size

        blurred_images = sorted([os.path.join(blurred_folder, img) for img in os.listdir(blurred_folder)])
        sharp_images = sorted([os.path.join(sharp_folder, img) for img in os.listdir(sharp_folder)])

        self.image_pairs = []
        for b, g in zip(blurred_images, sharp_images):
            self.image_pairs.append((b, g))

        random.shuffle(self.image_pairs)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        blurred_path, sharp_path = self.image_pairs[index]
        blurred_img = Image.open(blurred_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        blurred_img = self.transform(blurred_img)
        sharp_img = self.transform(sharp_img)
        return blurred_img, sharp_img
