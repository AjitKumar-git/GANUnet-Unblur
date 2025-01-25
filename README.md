# GANUnet-unblur: GAN-based Image Restoration Using U-Net

**GANUnet-unblur** is a project that uses Generative Adversarial Networks (GAN) for image restoration. Specifically, the generator is based on a **U-Net** architecture, which is widely used for tasks like image segmentation and restoration. The **discriminator** is based on a **Convolutional Neural Network (CNN)**, helping distinguish between real sharp images and the blurred images restored by the generator.

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Matplotlib
- PIL
- Other Python dependencies (listed in `requirements.txt`)

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AjitKumar-git/GANUnet-unblur.git
cd GANUnet-unblur
pip install -r requirements.txt
```

## Dataset Loader

The dataset loader handles different types of datasets, including loading paired images from a single folder, two separate folders, or real-world blurred images for training. The loader allows you to easily load and visualize the dataset.

### Available Dataset Types

1. **One Folder Dataset**  
   This type assumes that all images are located in a single folder, where each image has a corresponding sharp or ground truth version. It loads both blurred and sharp images from a single folder.

2. **Two Folder Dataset**  
   This type loads images from two separate folders. One folder contains the blurred images, while the other contains the corresponding sharp images. The images are paired by their filenames.

3. **Real Blur Dataset**  
   This type is used for real-world blurred datasets, where the dataset contains pairs of blurred and sharp images in different folders (e.g., Scenes where `blur` and `gt` folders are present).

### Using the Dataset Loader

You can use the dataset loader to load datasets of different types, and optionally visualize a few sample images. Below is an example of how to load and visualize the dataset.

#### Example Usage:

##### For RealBlur Dataset:

```bash
python main.py --mode train --dataset_type real_blur --root_dir "your_realblur_dataset_location(root directory only)" --image_size "256,256" --num_epochs 100
```

##### For One Folder Dataset:

```bash
python main.py --mode train --dataset_type one_folder --root_dir "your_dataset_location(root directory only)" --image_size "256,256" --num_epochs 100
```

##### For Two Folder Dataset:

```bash
python main.py --mode process --dataset_type two_folder --blurred_folder blur --sharp_folder sharp --image_size "256,256" --epoch 50
```

---
