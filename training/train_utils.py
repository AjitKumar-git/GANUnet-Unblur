import torch
import torch.nn.functional as F
import os

from models import save_checkpoint

def save_models(generator, discriminator, epoch, model_dir="models", save_interval=10):
    """
    Save the generator and discriminator models at specific intervals.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        epoch (int): The current epoch (0-indexed).
        model_dir (str): Directory to save the models. Default is "models".
        save_interval (int, optional): Save interval in epochs. Default is 10.
    """
    if save_interval is None or (epoch + 1) % save_interval == 0:
        if generator is not None:
            gen_path = os.path.join(model_dir, f"generator_epoch_{epoch+1}.pth")
            save_checkpoint(generator, gen_path)
        if discriminator is not None:
            disc_path = os.path.join(model_dir, f"discriminator_epoch_{epoch+1}.pth")
            save_checkpoint(discriminator, disc_path)


def train_discriminator(optimizer_D, discriminator, blurred_images, sharp_images, fake_images, adversarial_loss, device):
    optimizer_D.zero_grad()

    # Adversarial ground truths
    valid = torch.ones(sharp_images.size(0), 1, device=device)
    fake = torch.zeros(blurred_images.size(0), 1, device=device)

    # Measure discriminator's ability to classify real and fake images
    real_loss = adversarial_loss(discriminator(sharp_images), valid)
    fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    # Backward pass and optimize
    d_loss.backward()
    optimizer_D.step()

    return d_loss


def train_generator(optimizer_G, generator, discriminator, blurred_images, sharp_images, adversarial_loss, device):
    optimizer_G.zero_grad()

    # Generate a batch of images
    gen_images = generator(blurred_images)

    # Adversarial loss
    valid = torch.ones(sharp_images.size(0), 1, device=device)
    g_loss = adversarial_loss(discriminator(gen_images), valid)

    # Additional pixel-wise loss (e.g., L1 loss)
    pixel_loss = F.l1_loss(gen_images, sharp_images)
    total_g_loss = g_loss + 100 * pixel_loss  # Weighted combination

    # Backward pass and optimize
    total_g_loss.backward()
    optimizer_G.step()

    return total_g_loss
