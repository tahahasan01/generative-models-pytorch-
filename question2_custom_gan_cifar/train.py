# Fix OpenMP conflict (must be before torch imports)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid GUI-related native crashes when plotting from
# a background thread or on headless machines. This must happen before importing
# pyplot.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow disabling tqdm's monitor/thread via environment (useful to avoid threading issues on Windows)
DISABLE_TQDM = os.getenv('DISABLE_TQDM', '0') == '1'
import json
from datetime import datetime
import random
import copy
import argparse

from question2_custom_gan_cifar.models import Generator, SiameseDiscriminator, weights_init

# Hyperparameters (REBALANCED to fix mode collapse)
LATENT_DIM = 128
BATCH_SIZE = 64
NUM_EPOCHS = 200
LR_G = 0.0002  # Increased back to standard DCGAN value
LR_D = 0.0001  # REDUCED - discriminator was too strong
BETA1 = 0.5
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 5
GRADIENT_CLIP = 1.0  # Increased back for more flexibility
LABEL_SMOOTHING = 0.05  # REDUCED - was too aggressive with spectral norm
# Label value for 'real' when using smoothing (e.g., 0.95)
LABEL_REAL = 1.0 - LABEL_SMOOTHING

# Exponential Moving Average (EMA) for generator weights (recommended)
USE_EMA = True
EMA_DECAY = 0.999
# Note: EMA updates must also copy non-parameter buffers (BatchNorm running stats).

# Train discriminator and generator equally
D_STEPS = 1  # REDUCED from 2 - balanced training
D_TRAIN_FREQ = 1  # Train D every iteration
G_TRAIN_FREQ = 1  # Train G every iteration
GP_LAMBDA = 0.5  # REDUCED gradient penalty weight

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('generated_images', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cifar10_cats_dogs(augment=True):
    """
    Load CIFAR-10 dataset and filter only cats (class 3) and dogs (class 5)
    with optional data augmentation
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load full CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    
    # CIFAR-10 classes: cat=3, dog=5
    cat_dog_classes = [3, 5]
    
    # Filter training set for cats and dogs only
    train_indices = [i for i, (_, label) in enumerate(trainset) 
                     if label in cat_dog_classes]
    train_subset = Subset(trainset, train_indices)
    
    # Filter test set for cats and dogs only
    test_indices = [i for i, (_, label) in enumerate(testset) 
                    if label in cat_dog_classes]
    test_subset = Subset(testset, test_indices)
    
    print(f"Training samples (cats & dogs): {len(train_subset)}")
    print(f"Test samples (cats & dogs): {len(test_subset)}")
    
    return train_subset, test_subset


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    def __init__(self, optimizer, initial_lr, decay_factor=0.95, decay_interval=10):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        
    def step(self, epoch):
        if epoch > 0 and epoch % self.decay_interval == 0:
            new_lr = self.initial_lr * (self.decay_factor ** (epoch // self.decay_interval))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr
        return self.optimizer.param_groups[0]['lr']


def calculate_gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """Calculate gradient penalty for improved training stability (WGAN-GP style)"""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    
    # Create a second interpolation for the paired input
    alpha2 = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates2 = (alpha2 * real_imgs + (1 - alpha2) * fake_imgs).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates, interpolates2)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def add_noise_to_inputs(images, noise_std=0.1):
    """Add noise to discriminator inputs for stability"""
    noise = torch.randn_like(images) * noise_std
    return images + noise


def calculate_inception_score(generator, num_samples=5000, batch_size=100, splits=10):
    """Calculate a simplified quality metric"""
    generator.eval()
    scores = []
    
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_imgs = generator(z)
            # Simple metric: measure variance in pixel values (diversity proxy)
            scores.append(fake_imgs.std().item())
    
    generator.train()
    return np.mean(scores)


def save_training_info(epoch, g_losses, d_losses, lr_g, lr_d, quality_score):
    """Save training information to JSON"""
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'epoch': epoch,
        'generator_loss': float(g_losses[-1]),
        'discriminator_loss': float(d_losses[-1]),
        'learning_rate_G': lr_g,
        'learning_rate_D': lr_d,
        'quality_score': quality_score,
        'device': str(DEVICE)
    }
    
    log_file = f'logs/training_log_epoch_{epoch}.json'
    with open(log_file, 'w') as f:
        json.dump(info, f, indent=4)


def plot_advanced_metrics(g_losses, d_losses, quality_scores, epoch):
    """Plot comprehensive training metrics"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss curves
        axes[0, 0].plot(g_losses, label='Generator Loss', color='blue', linewidth=2)
        axes[0, 0].plot(d_losses, label='Discriminator Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Generator loss detail
        axes[0, 1].plot(g_losses, color='blue', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Generator Loss', fontsize=12)
        axes[0, 1].set_title('Generator Loss Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Discriminator loss detail
        axes[1, 0].plot(d_losses, color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Discriminator Loss', fontsize=12)
        axes[1, 0].set_title('Discriminator Loss Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Quality score
        axes[1, 1].plot(quality_scores, color='green', linewidth=2, marker='o', markersize=3)
        axes[1, 1].set_xlabel('Epoch (measured every 5 epochs)', fontsize=12)
        axes[1, 1].set_ylabel('Quality Score', fontsize=12)
        axes[1, 1].set_title('Generation Quality Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/training_metrics_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        # Catch plotting errors so they don't crash training (matplotlib C-level errors can
        # abort the whole process on some systems). Log the exception and continue.
        import traceback
        print(f"Warning: plotting failed at epoch {epoch}: {e}")
        traceback.print_exc()
        try:
            plt.close('all')
        except Exception:
            pass


def train_gan():
    """
    Train the GAN with Siamese discriminator (Advanced version)
    """
    # Set seed for reproducibility
    set_seed(42)
    
    print(f"Using device: {DEVICE}")
    print("="*70)
    
    # Load data with augmentation
    train_dataset, _ = load_cifar10_cats_dogs(augment=True)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=2,  # Reduced from 4 for better compatibility
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch for stability
    )
    
    # Initialize models
    generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    discriminator = SiameseDiscriminator().to(DEVICE)

    # Training history (initialize before potential resume so saved lists can be restored)
    g_losses = []
    d_losses = []
    quality_scores = []

    # We'll optionally load a checkpoint to resume training. If resuming,
    # avoid re-initializing weights (so load before any weight init call).
    start_epoch = 0
    resume_path = globals().get('RESUME_CHECKPOINT', None)
    if resume_path and os.path.exists(resume_path):
        try:
            print(f"Resuming training from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=DEVICE)
            # Load model weights
            if 'generator_state_dict' in ckpt and ckpt['generator_state_dict'] is not None:
                generator.load_state_dict(ckpt['generator_state_dict'])
            if 'discriminator_state_dict' in ckpt and ckpt['discriminator_state_dict'] is not None:
                discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            # Restore losses/metrics if present
            if 'g_losses' in ckpt and ckpt['g_losses']:
                g_losses = ckpt.get('g_losses', g_losses)
            if 'd_losses' in ckpt and ckpt['d_losses']:
                d_losses = ckpt.get('d_losses', d_losses)
            if 'quality_scores' in ckpt and ckpt['quality_scores']:
                quality_scores = ckpt.get('quality_scores', quality_scores)
            # Determine start epoch (saved epoch is zero-indexed)
            if 'epoch' in ckpt and ckpt['epoch'] is not None:
                start_epoch = int(ckpt['epoch']) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint {resume_path}: {e}")
    else:
        # Apply weight initialization for a fresh start
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # EMA generator (optional)
    ema_generator = None
    if USE_EMA:
        ema_generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
        ema_generator.load_state_dict(generator.state_dict())
        # EMA generator is only used for evaluation/sampling
        for p in ema_generator.parameters():
            p.requires_grad_(False)
        ema_generator.eval()
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # If resuming, restore optimizer states (best-effort)
    if resume_path and os.path.exists(resume_path):
        try:
            if 'optimizer_G_state_dict' in ckpt and ckpt['optimizer_G_state_dict'] is not None:
                optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
            if 'optimizer_D_state_dict' in ckpt and ckpt['optimizer_D_state_dict'] is not None:
                optimizer_D.load_state_dict(ckpt['optimizer_D_state_dict'])
        except Exception as e:
            print(f"Warning: failed to restore optimizer state from checkpoint: {e}")
    
    # Learning rate schedulers
    scheduler_G = LearningRateScheduler(optimizer_G, LR_G, decay_factor=0.95, decay_interval=20)
    scheduler_D = LearningRateScheduler(optimizer_D, LR_D, decay_factor=0.95, decay_interval=20)
    
    # Early stopping (REDUCED patience to catch collapse earlier)
    early_stopping = EarlyStopping(patience=15, min_delta=0.01)
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM).to(DEVICE)
    
    # (moved earlier to allow resume loading)
    
    # Best model tracking
    best_g_loss = float('inf')
    
    print("\nStarting Advanced Training with:")
    print(f"  [*] Data Augmentation")
    print(f"  [*] Gradient Clipping")
    print(f"  [*] Learning Rate Scheduling")
    print(f"  [*] Early Stopping")
    print(f"  [*] Input Noise for Stability")
    print(f"  [*] Quality Metrics Tracking")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        epoch_g_loss = 0
        epoch_d_loss = 0
        # Always show epoch start in stdout so users running with DISABLE_TQDM
        # can still see progress. Also keep running values for intermittent
        # batch-level prints when tqdm is disabled.
        print(f"\n==> Starting epoch {epoch+1}/{NUM_EPOCHS}")
        last_g_loss = 0.0
        last_d_loss = 0.0
        
        # Adjust noise level over time (reduce as training progresses)
        noise_std = max(0.05, 0.2 * (1 - epoch / NUM_EPOCHS))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=DISABLE_TQDM)
        
        for i, (real_imgs, _) in enumerate(pbar):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            
            # =====================
            # Train Discriminator
            # =====================
            if i % D_TRAIN_FREQ == 0:
                # Run multiple discriminator updates per generator update (D_STEPS)
                for d_iter in range(D_STEPS):
                    optimizer_D.zero_grad()

                    # Add noise to real images for stability
                    real_imgs_noisy = add_noise_to_inputs(real_imgs, noise_std)

                    # Generate fake images
                    z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
                    fake_imgs = generator(z).detach()
                    fake_imgs_noisy = add_noise_to_inputs(fake_imgs, noise_std)

                    # Compute similarity score for (fake, real) pair
                    # With Tanh output: negative = dissimilar, positive = similar
                    fake_real_similarity = discriminator(fake_imgs_noisy, real_imgs_noisy)

                    # Compute similarity for (real, real) pair to help D learn similarity
                    idx = torch.randperm(batch_size).to(DEVICE)
                    real_imgs_shuffled = real_imgs_noisy[idx]
                    real_real_similarity = discriminator(real_imgs_noisy, real_imgs_shuffled)

                    # Add gradient penalty for stability
                    gp = calculate_gradient_penalty(discriminator, real_imgs, fake_imgs, DEVICE)

                    # MSE Loss: D wants fake-real → -1 (dissimilar) and real-real → LABEL_REAL (smoothed)
                    d_loss_fake = torch.mean((fake_real_similarity + 1) ** 2)  # target -1
                    d_loss_real = torch.mean((real_real_similarity - LABEL_REAL) ** 2)
                    d_loss = d_loss_fake + d_loss_real + GP_LAMBDA * gp

                    d_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRADIENT_CLIP)

                    optimizer_D.step()

                    epoch_d_loss += d_loss.item()
            
            # ==================
            # Train Generator
            # ==================
            if i % G_TRAIN_FREQ == 0:
                optimizer_G.zero_grad()
                
                # Generate new fake images
                z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
                fake_imgs = generator(z)
                
                # Compute similarity score for (fake, real) pair
                # Note: reusing real_imgs from batch (D already saw these); could sample fresh batch for G
                fake_real_similarity = discriminator(fake_imgs, real_imgs)
                
                # NEW LOSS: G wants fake-real to be positive (similar, like real-real)
                # Target similarity of +1 (like real images)
                g_loss = torch.mean((fake_real_similarity - 1) ** 2)
                
                g_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRADIENT_CLIP)
                
                optimizer_G.step()

                # Update EMA weights after generator update
                if USE_EMA and ema_generator is not None:
                    with torch.no_grad():
                        # Update EMA parameters (by name to be safe)
                        g_params = dict(generator.named_parameters())
                        for name, ema_p in ema_generator.named_parameters():
                            if name in g_params:
                                ema_p.data.mul_(EMA_DECAY).add_(g_params[name].data, alpha=1.0 - EMA_DECAY)
                        # Copy buffers (BatchNorm running_mean/var etc.) directly from generator
                        g_buffers = dict(generator.named_buffers())
                        for name, ema_b in ema_generator.named_buffers():
                            if name in g_buffers:
                                ema_b.data.copy_(g_buffers[name].data)
                
                epoch_g_loss += g_loss.item()
            
            # Update progress bar / occasional stdout status
            try:
                last_g_loss = g_loss.item()
            except Exception:
                pass
            try:
                last_d_loss = d_loss.item()
            except Exception:
                pass

            # If tqdm is active, keep setting the postfix. If disabled (e.g. on
            # Windows with DISABLE_TQDM=1), print occasional status lines so the
            # user can still see progress in the terminal.
            if not DISABLE_TQDM:
                pbar.set_postfix({
                    'G_loss': f'{last_g_loss:.4f}',
                    'D_loss': f'{last_d_loss:.4f}',
                    'LR_G': f'{optimizer_G.param_groups[0]["lr"]:.6f}'
                })
            else:
                # Print every 50 batches (adjustable) to avoid flooding output
                if i % 50 == 0:
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {i+1}/{len(dataloader)} | G_loss: {last_g_loss:.4f} | D_loss: {last_d_loss:.4f}")
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Update learning rates
        lr_g = scheduler_G.step(epoch)
        lr_d = scheduler_D.step(epoch)
        
        # Calculate quality score every 5 epochs (can be disabled)
        quality_score = 0
        if not globals().get('DISABLE_PLOTTING', False) and (epoch + 1) % 5 == 0:
            quality_score = calculate_inception_score(generator)
            quality_scores.append(quality_score)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | Quality: {quality_score:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f}")
        
        # Save training info every epoch (so metrics are available per-epoch)
        try:
            save_training_info(epoch + 1, g_losses, d_losses, lr_g, lr_d, quality_score)
        except Exception as e:
            print(f"Warning: failed to write training info for epoch {epoch+1}: {e}")
        
        # Check for best model
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(generator.state_dict(), 'checkpoints/generator_best.pth')
            torch.save(discriminator.state_dict(), 'checkpoints/discriminator_best.pth')
            # Save EMA generator if enabled
            if USE_EMA and ema_generator is not None:
                torch.save(ema_generator.state_dict(), 'checkpoints/generator_best_ema.pth')
            print(f"  [*] New best model saved! (G_loss: {best_g_loss:.4f})")
        
        # Early stopping check
        early_stopping(avg_g_loss)
        if early_stopping.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save generated images every epoch (keeps lightweight per-epoch previews)
        try:
            with torch.no_grad():
                # Prefer EMA generator for visualization if available
                if USE_EMA and ema_generator is not None:
                    ema_generator.eval()  # Ensure eval mode (BN uses running stats)
                    fake = ema_generator(fixed_noise)
                else:
                    generator.eval()
                    fake = generator(fixed_noise)
                    generator.train()  # restore train mode
                fake = (fake + 1) / 2  # Denormalize
                sample_path = f'generated_images/epoch_{epoch+1:03d}.png'
                save_image(
                    fake,
                    sample_path,
                    nrow=8,
                    normalize=False
                )

            # Also copy the latest sample to results for quick preview access
            try:
                import shutil
                shutil.copyfile(sample_path, 'results/sample_preview_latest.png')
            except Exception as e:
                print(f"Warning: could not copy latest sample to results/: {e}")
        except Exception as e:
            print(f"Warning: failed to generate/save sample for epoch {epoch+1}: {e}")

        # Save full checkpoints only at SAVE_INTERVAL (to avoid excessive disk usage)
        if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'ema_generator_state_dict': ema_generator.state_dict() if (USE_EMA and ema_generator is not None) else None,
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'quality_scores': quality_scores,
                'best_g_loss': best_g_loss,
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

            print(f"  [*] Saved checkpoint for epoch {epoch+1}")

            # Plot advanced metrics (skip if plotting disabled)
            if (not globals().get('DISABLE_PLOTTING', False)):
                try:
                    plot_advanced_metrics(g_losses, d_losses, quality_scores, epoch + 1)
                except Exception as e:
                    print(f"Warning: plotting failed at epoch {epoch+1}: {e}")
    
    # Save final models
    torch.save(generator.state_dict(), 'checkpoints/generator_final.pth')
    torch.save(discriminator.state_dict(), 'checkpoints/discriminator_final.pth')
    if USE_EMA and ema_generator is not None:
        torch.save(ema_generator.state_dict(), 'checkpoints/generator_final_ema.pth')
    
    # Final metrics plot
    if quality_scores:
        plot_advanced_metrics(g_losses, d_losses, quality_scores, NUM_EPOCHS)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print(f"[*] Best G_loss: {best_g_loss:.4f}")
    print(f"[*] Final models saved to 'checkpoints/' directory")
    print(f"[*] Generated images saved to 'generated_images/' directory")
    print(f"[*] Training logs saved to 'logs/' directory")
    print(f"[*] Metrics plots saved to 'results/' directory")
    print("="*70)


def generate_samples(num_samples=64, checkpoint_path='checkpoints/generator_best.pth'):
    """Generate samples using the best trained generator"""
    print(f"\nGenerating {num_samples} samples...")
    
    # Prefer EMA best checkpoint if available when using default path
    if checkpoint_path == 'checkpoints/generator_best.pth' and USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
        checkpoint_path = 'checkpoints/generator_best_ema.pth'

    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        print(f"  Using final model instead...")
        # Prefer EMA best if available
        if USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
            checkpoint_path = 'checkpoints/generator_best_ema.pth'
        else:
            checkpoint_path = 'checkpoints/generator_final.pth'
    
    generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM).to(DEVICE)
        fake_imgs = generator(z)
        fake_imgs = (fake_imgs + 1) / 2  # Denormalize
        
        save_image(
            fake_imgs,
            'results/final_samples.png',
            nrow=8,
            normalize=False
        )
    
    print(f"[*] Generated samples saved to 'results/final_samples.png'")


def visualize_interpolation(checkpoint_path='checkpoints/generator_best.pth', num_interpolations=10):
    """Visualize interpolation between two random points in latent space"""
    print(f"\nGenerating latent space interpolation...")
    
    # Prefer EMA best checkpoint if available when using default path
    if checkpoint_path == 'checkpoints/generator_best.pth' and USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
        checkpoint_path = 'checkpoints/generator_best_ema.pth'

    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        print(f"  Using final model instead...")
        if USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
            checkpoint_path = 'checkpoints/generator_best_ema.pth'
        else:
            checkpoint_path = 'checkpoints/generator_final.pth'
    
    generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    generator.eval()
    
    with torch.no_grad():
        # Generate two random latent vectors
        z1 = torch.randn(1, LATENT_DIM).to(DEVICE)
        z2 = torch.randn(1, LATENT_DIM).to(DEVICE)
        
        # Interpolate between them
        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            fake_img = generator(z_interp)
            fake_img = (fake_img + 1) / 2  # Denormalize
            interpolations.append(fake_img)
        
        interpolations = torch.cat(interpolations, dim=0)
        save_image(
            interpolations,
            'results/latent_interpolation.png',
            nrow=num_interpolations,
            normalize=False
        )
    
    print(f"[*] Interpolation saved to 'results/latent_interpolation.png'")


def generate_grid_samples(checkpoint_path='checkpoints/generator_best.pth'):
    """Generate a large grid of diverse samples"""
    print(f"\nGenerating diverse sample grid...")
    
    # Prefer EMA best checkpoint if available when using default path
    if checkpoint_path == 'checkpoints/generator_best.pth' and USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
        checkpoint_path = 'checkpoints/generator_best_ema.pth'

    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        print(f"  Using final model instead...")
        if USE_EMA and os.path.exists('checkpoints/generator_best_ema.pth'):
            checkpoint_path = 'checkpoints/generator_best_ema.pth'
        else:
            checkpoint_path = 'checkpoints/generator_final.pth'
    
    generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(100, LATENT_DIM).to(DEVICE)
        fake_imgs = generator(z)
        fake_imgs = (fake_imgs + 1) / 2
        
        save_image(
            fake_imgs,
            'results/diverse_samples_grid.png',
            nrow=10,
            normalize=False
        )
    
    print(f"[*] Diverse sample grid saved to 'results/diverse_samples_grid.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CIFAR pairwise GAN (Q2)')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=None, help='training batch size (default: 64)')
    parser.add_argument('--no-ema', action='store_true', help='disable EMA generator')
    parser.add_argument('--skip-eval', action='store_true', help='skip final sample generation/eval')
    parser.add_argument('--no-plot', action='store_true', help='disable runtime plotting and metrics to avoid native crashes')
    parser.add_argument('--data_root', type=str, default='./data', help='path to CIFAR-10 data root')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory for checkpoints/images (ignored for now, uses default paths)')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint .pth file to resume training from')
    args = parser.parse_args()

    # Override module-level variables from CLI args if provided
    if args.epochs is not None:
        globals()['NUM_EPOCHS'] = args.epochs
    if args.batch_size is not None:
        globals()['BATCH_SIZE'] = args.batch_size
    if args.no_ema:
        globals()['USE_EMA'] = False
    if args.no_plot:
        # Disable runtime metric calculation and plotting to avoid native library crashes
        globals()['DISABLE_PLOTTING'] = True
    else:
        globals()['DISABLE_PLOTTING'] = False

    # Make resume path available to train_gan via globals()
    globals()['RESUME_CHECKPOINT'] = args.resume

    print("\n" + "="*70)
    print(f"    ADVANCED GAN TRAINING WITH SIAMESE DISCRIMINATOR (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, EMA={'on' if USE_EMA else 'off'})")
    print("="*70 + "\n")

    # Train the GAN (supports resuming from a checkpoint via --resume)
    train_gan()

    if not args.skip_eval:
        # Generate final samples using best model (prefers EMA if available)
        generate_samples(num_samples=64)
        # Visualize latent space interpolation
        visualize_interpolation(num_interpolations=10)
        # Generate diverse sample grid
        generate_grid_samples()

    print("\n" + "="*70)
    print("All tasks completed successfully!")
    print("="*70 + "\n")