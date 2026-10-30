import os
import shutil
import pytest
import torch
from models import Generator, Discriminator, save_checkpoint, load_checkpoint, load_trained_models
from training.train_utils import save_models

@pytest.fixture
def tmp_model_dir(tmp_path):
    dir_path = tmp_path / "models"
    yield str(dir_path)
    if os.path.exists(str(dir_path)):
        shutil.rmtree(str(dir_path))

def test_save_and_load_checkpoint(tmp_model_dir):
    gen = Generator()
    filepath = os.path.join(tmp_model_dir, "gen.pth")
    save_checkpoint(gen, filepath)
    assert os.path.exists(filepath)

    gen_loaded = Generator()
    load_checkpoint(gen_loaded, filepath, device="cpu")
    # Verify weights match
    for p1, p2 in zip(gen.parameters(), gen_loaded.parameters()):
        assert torch.equal(p1, p2)

def test_save_models(tmp_model_dir):
    gen = Generator()
    disc = Discriminator()
    save_models(gen, disc, epoch=9, model_dir=tmp_model_dir, save_interval=10)

    assert os.path.exists(os.path.join(tmp_model_dir, "generator_epoch_10.pth"))
    assert os.path.exists(os.path.join(tmp_model_dir, "discriminator_epoch_10.pth"))

def test_load_trained_models_generator_only(tmp_model_dir):
    gen = Generator()
    disc = Discriminator()
    save_models(gen, disc, epoch=9, model_dir=tmp_model_dir, save_interval=10)

    gen_target = Generator()
    load_trained_models(generator=gen_target, discriminator=None, model_dir=tmp_model_dir, epoch=10, device="cpu")

    for p1, p2 in zip(gen.parameters(), gen_target.parameters()):
        assert torch.equal(p1, p2)
