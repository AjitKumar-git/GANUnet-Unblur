import os
import shutil
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import Generator, Discriminator
from training.train_utils import train_discriminator, train_generator
from visualization.display_utils import save_generated_images
from main import process_images

@pytest.fixture
def tmp_output_dir(tmp_path):
    dir_path = tmp_path / "generated"
    yield str(dir_path)
    if os.path.exists(str(dir_path)):
        shutil.rmtree(str(dir_path))

def test_training_steps():
    device = torch.device("cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    adversarial_loss = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    blurred = torch.randn(2, 3, 128, 128)
    sharp = torch.randn(2, 3, 128, 128)

    fake = generator(blurred)
    d_loss = train_discriminator(optimizer_D, discriminator, blurred, sharp, fake, adversarial_loss, device)
    assert d_loss.item() >= 0

    g_loss = train_generator(optimizer_G, generator, discriminator, blurred, sharp, adversarial_loss, device)
    assert g_loss.item() >= 0

def test_process_images_and_save(tmp_output_dir):
    device = torch.device("cpu")
    generator = Generator().to(device)
    blurred = torch.randn(4, 3, 128, 128)
    sharp = torch.randn(4, 3, 128, 128)

    dataset = TensorDataset(blurred, sharp)
    dataloader = DataLoader(dataset, batch_size=2)

    process_images(generator, dataloader, device, output_dir=tmp_output_dir)
    assert os.path.exists(os.path.join(tmp_output_dir, "generated_epoch_1.png"))
    assert os.path.exists(os.path.join(tmp_output_dir, "generated_epoch_2.png"))
