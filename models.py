import os
import torch
import torch.nn as nn

def save_checkpoint(model, filepath):
    """
    Saves a PyTorch model's state dictionary to filepath.
    Automatically creates directories if they don't exist.
    """
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model checkpoint saved to {filepath}")


def load_checkpoint(model, filepath, device='cpu'):
    """
    Loads weights from filepath into the given model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded checkpoint from {filepath}")
    return model


def load_trained_models(generator=None, discriminator=None, model_dir="models", epoch=None, gen_path=None, disc_path=None, device="cpu"):
    """
    Harmonized model loader for trained generator and/or discriminator models.
    Supports loading by epoch number or explicit checkpoint paths.
    """
    if gen_path is None and epoch is not None and generator is not None:
        gen_path = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
    if disc_path is None and epoch is not None and discriminator is not None:
        disc_path = os.path.join(model_dir, f"discriminator_epoch_{epoch}.pth")

    if generator is not None:
        if gen_path and os.path.exists(gen_path):
            load_checkpoint(generator, gen_path, device=device)
        elif gen_path:
            raise FileNotFoundError(f"Generator model checkpoint not found: {gen_path}")

    if discriminator is not None:
        if disc_path and os.path.exists(disc_path):
            load_checkpoint(discriminator, disc_path, device=device)
        elif disc_path:
            print(f"Warning: Discriminator checkpoint not found at {disc_path}")

    return generator, discriminator


# Generator model
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(Generator, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.double_conv(feature * 2, feature))

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 128, 128)
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 64, 64)
            nn.BatchNorm2d(64, momentum=0.82),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 32, 32)
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> (256, 16, 16)
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # -> (512, 8, 8)
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # -> (1024, 4, 4)
            nn.BatchNorm2d(1024, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
