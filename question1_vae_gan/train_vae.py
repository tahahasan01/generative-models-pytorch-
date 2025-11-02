import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from models import SimpleVAE
from data_utils import SignatureDataset
from torchvision.utils import save_image
from scripts.eval_metrics import mse, psnr, ssim_index
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import csv


def get_dataset(root, augment=False):
    if augment:
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((64,64), scale=(0.8,1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
        return SignatureDataset(root, transform=tf)
    return SignatureDataset(root)


def tensor_to_uint8_images(tensor):
    # tensor: (N, C, H, W), assumed in [0,1]
    arr = (tensor.detach().cpu().numpy() * 255.0).astype('uint8')
    arr = arr.transpose(0, 2, 3, 1)
    if arr.shape[3] == 1:
        arr = np.repeat(arr, 3, axis=3)
    return arr


def train(args):
    ds = get_dataset(args.data_root, augment=args.augment)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVAE(z_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if getattr(args, 'amp', False) and device.type == 'cuda' else None
    out_dir = os.path.join('outputs', 'question1_vae')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'epoch_loss', 'mse', 'psnr', 'ssim'])
    for epoch in range(args.epochs):
        model.train()
        total = 0
        for xb in loader:
            xb = xb.to(device)
            if scaler is not None:
                with autocast():
                    xrec, mu, logvar = model(xb)
                    rec_loss = torch.nn.functional.mse_loss(xrec, xb)
                    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = rec_loss + args.beta * kld
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item()
            else:
                xrec, mu, logvar = model(xb)
                rec_loss = torch.nn.functional.mse_loss(xrec, xb)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + args.beta * kld
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
        print(f'Epoch {epoch} loss {total/len(loader):.4f}')
        # evaluation on first batch of loader
        model.eval()
        with torch.no_grad():
            xb = next(iter(loader)).to(device)
            xrec, _, _ = model(xb)
            # save a grid of originals and reconstructions
            save_image(xb[:8], os.path.join(out_dir, f'orig_epoch{epoch}.png'))
            save_image(xrec[:8], os.path.join(out_dir, f'rec_epoch{epoch}.png'))
            # compute metrics
            orig_np = (xb[:8].cpu().numpy() * 255).astype('uint8').transpose(0,2,3,1)
            rec_np = (xrec[:8].cpu().numpy() * 255).astype('uint8').transpose(0,2,3,1)
            if orig_np.shape[3] == 1:
                orig_np = np.repeat(orig_np, 3, axis=3)
                rec_np = np.repeat(rec_np, 3, axis=3)
            cur_mse = mse(orig_np, rec_np)
            cur_psnr = psnr(orig_np, rec_np)
            cur_ssim = ssim_index(orig_np, rec_np)
            print('Eval MSE', cur_mse, 'PSNR', cur_psnr, 'SSIM', cur_ssim)
            # append metrics to CSV
            try:
                with open(metrics_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, total / max(1, len(loader)), float(cur_mse), float(cur_psnr), float(cur_ssim)])
            except Exception as e:
                print('Failed writing metrics csv:', e)
        # checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, os.path.join(ckpt_dir, f'vae_epoch{epoch}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (automatic)')
    args = parser.parse_args()
    train(args)
