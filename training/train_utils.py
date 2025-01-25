import torch
import torch.nn.functional as F
import os

def save_models(generator, discriminator, epoch, model_dir="models"):
    """
    Save the generator and discriminator models at specific intervals.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        epoch (int): The current epoch.
        model_dir (str): Directory to save the models. Default is "models".
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save models at intervals (every 50 epochs by default)
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch+1}.pth"))
        print(f"Models saved at epoch {epoch + 1}")


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
