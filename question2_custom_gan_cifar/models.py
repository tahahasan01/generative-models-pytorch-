import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    """Generator network for CIFAR-10 image generation"""
    def __init__(self, latent_dim=100, img_channels=3):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Input: latent vector of size latent_dim
        # Output: 3x32x32 image
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 32 x 32
        )
    
    def forward(self, z):
        # Reshape latent vector to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.model(z)


class SiameseDiscriminator(nn.Module):
    """
    Siamese-style discriminator that computes similarity between two images.
    Takes two images as input and outputs a similarity score.
    """
    def __init__(self, img_channels=3, use_spectral_norm=True):
        super(SiameseDiscriminator, self).__init__()

        # Shared feature extractor (twin networks)
        # Apply spectral normalization to conv layers for stability (if enabled)
        conv = spectral_norm if use_spectral_norm else lambda x: x

        self.feature_extractor = nn.Sequential(
            # Input: 3 x 32 x 32
            conv(nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 16 x 16

            conv(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 8 x 8

            conv(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 4 x 4

            conv(nn.Conv2d(256, 512, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 1 x 1
        )

        # Similarity computation network
        # Takes concatenated features from both images
        # Removed Dropout to preserve feature information for discriminator
        self.similarity_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Tanh()  # Bound output to [-1, 1] for stability
        )
    
    def forward_one(self, x):
        """Extract features for one image"""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features
    
    def forward(self, img1, img2):
        """
        Compute similarity score between two images.
        Lower score = more similar (minimize for generator)
        Higher score = less similar (maximize for discriminator with fake-real pairs)
        """
        # Extract features from both images using shared network
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        
        # Concatenate features
        combined = torch.cat([feat1, feat2], dim=1)
        
        # Compute similarity score
        similarity = self.similarity_net(combined)
        
        return similarity


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Generator
    gen = Generator(latent_dim=100).to(device)
    gen.apply(weights_init)
    z = torch.randn(8, 100).to(device)
    fake_imgs = gen(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    # Test Siamese Discriminator
    disc = SiameseDiscriminator().to(device)
    disc.apply(weights_init)
    real_imgs = torch.randn(8, 3, 32, 32).to(device)
    similarity_score = disc(fake_imgs, real_imgs)
    print(f"Discriminator similarity score shape: {similarity_score.shape}")
    print(f"Sample similarity scores: {similarity_score.squeeze()[:4]}")