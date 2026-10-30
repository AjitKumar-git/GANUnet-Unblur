import pytest
import torch
from models import Generator, Discriminator

def test_generator_forward():
    generator = Generator()
    generator.eval()
    for size in [64, 128, 256]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = generator(x)
        assert out.shape == (2, 3, size, size), f"Expected shape {(2, 3, size, size)}, got {out.shape}"

def test_discriminator_forward():
    discriminator = Discriminator()
    discriminator.eval()
    for size in [64, 128, 256, 512]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = discriminator(x)
        assert out.shape == (2, 1), f"Expected shape {(2, 1)}, got {out.shape}"
