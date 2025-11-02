import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from models import SimpleGenerator, SimpleDiscriminator
from data_utils import SignatureDataset
from torchvision.utils import save_image
from scripts.eval_metrics import mse, psnr, ssim_index
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms


def get_dataset(root, augment=False):
    if augment:
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((64,64), scale=(0.8,1.0)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ])
        return SignatureDataset(root, transform=tf)
    return SignatureDataset(root)


def train(args):
    ds = get_dataset(args.data_root, augment=args.augment)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = SimpleGenerator(z_dim=128).to(device)
    D = SimpleDiscriminator().to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    scaler = GradScaler() if getattr(args, 'amp', False) and device.type == 'cuda' else None
    # EMA for generator
    ema = None
    ema_decay = float(getattr(args, 'ema_decay', 0.999)) if getattr(args, 'ema', False) else None
    if getattr(args, 'ema', False):
        ema = {name: param.detach().clone().to('cpu') for name, param in G.state_dict().items()}
    out_dir = os.path.join('outputs', 'question1_gan')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)


def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp).view(-1)
    grad = torch.autograd.grad(outputs=d_interp.sum(), inputs=interp, create_graph=True)[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
    # WGAN-GP training
    n_critic = args.n_critic
    lambda_gp = args.lambda_gp
    for epoch in range(args.epochs):
        for i, xb in enumerate(loader):
            xb = xb.to(device)
            bs = xb.size(0)

            # train discriminator n_critic times
            for _ in range(n_critic):
                z = torch.randn(bs, 128, device=device)
                if scaler is not None:
                    with autocast():
                        fake = G(z).detach()
                        d_real = D(xb).view(-1).mean()
                        d_fake = D(fake).view(-1).mean()
                        gp = gradient_penalty(D, xb, fake, device)
                        d_loss = d_fake - d_real + lambda_gp * gp
                    d_opt.zero_grad()
                    scaler.scale(d_loss).backward()
                    scaler.step(d_opt)
                    scaler.update()
                else:
                    fake = G(z).detach()
                    d_real = D(xb).view(-1).mean()
                    d_fake = D(fake).view(-1).mean()
                    gp = gradient_penalty(D, xb, fake, device)
                    d_loss = d_fake - d_real + lambda_gp * gp
                    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # train generator
            z = torch.randn(bs, 128, device=device)
            if scaler is not None:
                with autocast():
                    fake = G(z)
                    g_loss = -D(fake).view(-1).mean()
                g_opt.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.step(g_opt)
                scaler.update()
            else:
                fake = G(z)
                g_loss = -D(fake).view(-1).mean()
                g_opt.zero_grad(); g_loss.backward(); g_opt.step()

            # update EMA
            if ema is not None:
                # move current params to cpu and update
                for k, v in G.state_dict().items():
                    v_cpu = v.detach().cpu()
                    ema[k] = ema[k] * ema_decay + (1.0 - ema_decay) * v_cpu

        print(f'Epoch {epoch} g_loss {g_loss.item():.4f} d_loss {d_loss.item():.4f} gp {gp.item():.4f}')
        # save generated samples and checkpoint
        with torch.no_grad():
            z = torch.randn(16, 128, device=device)
            samples = G(z)
            save_image(samples, os.path.join(out_dir, f'samples_epoch{epoch}.png'), nrow=4)
        torch.save({'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict(), 'g_opt': g_opt.state_dict(), 'd_opt': d_opt.state_dict()}, os.path.join(ckpt_dir, f'gan_epoch{epoch}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (automatic)')
    parser.add_argument('--ema', action='store_true', help='Enable EMA for generator')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    parser.add_argument('--eval_fid_samples', type=int, default=128, help='Samples used for FID evaluation')
    args = parser.parse_args()
    train(args)
