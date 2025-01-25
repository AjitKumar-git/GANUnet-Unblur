import torch
import argparse
from torch.utils.data import DataLoader
from models import Generator, Discriminator  # Assuming your models are in a 'models.py' file
from training.train_utils import train_discriminator, train_generator, save_models  # Import functions from train_utils.py
from visualization.display_utils import save_generated_images  # Import the display function
from datasets.dataset_loader import load_dataset, visualize_dataset  # Import from dataset_loader.py

def train_gan(generator, discriminator, dataloader, num_epochs, lr, beta1, beta2, device, model_dir="models"):
    # Define the loss function
    adversarial_loss = torch.nn.BCELoss()

    # Optimizers for the generator and discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Start training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            blurred_images = batch[0].to(device)
            sharp_images = batch[1].to(device)

            fake_images = generator(blurred_images)

            # Train discriminator
            d_loss = train_discriminator(optimizer_D, discriminator, blurred_images, sharp_images, fake_images, adversarial_loss, device)

            # Train generator
            total_g_loss = train_generator(optimizer_G, generator, discriminator, blurred_images, sharp_images, adversarial_loss, device)

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {total_g_loss.item():.4f}"
                )

        # Save generated images and models for every epoch
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, blurred_images, epoch, device)
            save_models(generator, discriminator, epoch, model_dir)

    print("Training completed!")

def process_images(generator, dataloader, device):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            blurred_images = batch[0].to(device)
            
            # Generate images using the trained generator
            generated_images = generator(blurred_images)

            # Save the processed images
            save_generated_images(generator, blurred_images, i, device)

    print("Image processing completed!")

def load_trained_models(generator, discriminator, model_dir, epoch):
    # Load the trained model states
    generator.load_state_dict(torch.load(f"{model_dir}/generator_epoch_{epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"{model_dir}/discriminator_epoch_{epoch}.pth"))
    print(f"Models loaded for epoch {epoch}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or process images with a GAN")
    parser.add_argument('--mode', choices=['train', 'process'], required=True, help="Mode of operation: train or process")
    parser.add_argument('--dataset_type', choices=['one_folder', 'two_folder', 'real_blur'], required=True, help="Dataset type")
    parser.add_argument('--root_dir', required=True, help="Root directory for the dataset")
    parser.add_argument('--blurred_folder', help="Folder containing blurred images (for 'two_folder' and 'real_blur')")
    parser.add_argument('--sharp_folder', help="Folder containing sharp images (for 'two_folder')")
    parser.add_argument('--gt_folder', help="Folder containing ground truth images (for 'real_blur')")
    parser.add_argument('--model_dir', default='models', help="Directory where models are saved")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--image_size', type=lambda s: tuple(map(int, s.split(','))), default=(256, 256), help="Image size as a tuple (e.g., 256,256)")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--beta1', type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument('--epoch', type=int, help="Epoch to load for image processing mode")

    # Parse arguments
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset(
        dataset_type=args.dataset_type,
        root_dir=args.root_dir,
        blurred_folder=args.blurred_folder,
        sharp_folder=args.sharp_folder,
        gt_folder=args.gt_folder,
        image_size=args.image_size
    )

    # Visualize dataset if needed (optional, for debugging)
    visualize_dataset(dataset, num_samples=5)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if args.mode == 'train':
        # Train the GAN
        train_gan(generator, discriminator, dataloader, args.num_epochs, args.lr, args.beta1, args.beta2, device, args.model_dir)

    elif args.mode == 'process':
        # Load the trained models
        if args.epoch is None:
            print("Error: --epoch must be provided in 'process' mode")
        else:
            load_trained_models(generator, discriminator, args.model_dir, args.epoch)
            # Process images using the trained generator
            process_images(generator, dataloader, device)
