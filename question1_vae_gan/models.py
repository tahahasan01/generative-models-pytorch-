import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, z_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, z_dim)

        self.fc_dec = nn.Linear(z_dim, 64 * 16 * 16)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparam(mu, logvar)
        h2 = self.fc_dec(z)
        xrec = self.dec(h2)
        return xrec, mu, logvar


class SimpleGenerator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
