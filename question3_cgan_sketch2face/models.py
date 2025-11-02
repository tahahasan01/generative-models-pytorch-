"""
Conditional GAN for Sketch-to-Face Generation
Based on: Conditional Generative Adversarial Nets (Mirza & Osindero, 2014)

Complete implementation with training and user-friendly inference interface
that accepts ANY user-provided sketch and generates realistic face images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# ============================================
# GENERATOR NETWORK
# ============================================
class Generator(nn.Module):
    """
    Generator conditioned on sketch input
    Architecture: Noise + Sketch → Joint Representation → Generated Face
    """
    def __init__(self, noise_dim=100, sketch_channels=1, output_channels=3, img_size=64):
        super(Generator, self).__init__()
        
        # Noise processing branch (maps to hidden layer as per paper)
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        
        # Sketch encoding branch (condition y mapped to hidden representation)
        self.sketch_encoder = nn.Sequential(
            nn.Conv2d(sketch_channels, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, 4, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        # Combined representation (joint hidden layer from paper)
        # 512 * 4 * 4 = 8192 from sketch + 256 from noise = 8448
        self.fc_combined = nn.Sequential(
            nn.Linear(8192 + 256, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(True)
        )
        
        # Decoder to generate face image (output layer)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise, sketch):
        """
        Forward pass: G(z|y)
        Args:
            noise: Random noise vector [batch, noise_dim]
            sketch: Sketch condition [batch, 1, H, W]
        Returns:
            Generated face image [batch, 3, H, W]
        """
        # Process noise
        noise_features = self.noise_fc(noise)  # [B, 256]
        
        # Process sketch condition
        sketch_features = self.sketch_encoder(sketch)  # [B, 512, 4, 4]
        sketch_features = sketch_features.view(sketch_features.size(0), -1)  # [B, 8192]
        
        # Combine both inputs (joint representation as per paper)
        combined = torch.cat([noise_features, sketch_features], dim=1)  # [B, 8448]
        combined = self.fc_combined(combined)  # [B, 8192]
        
        # Reshape for decoder
        combined = combined.view(-1, 512, 4, 4)
        
        # Generate face image
        output = self.decoder(combined)
        return output


# ============================================
# DISCRIMINATOR NETWORK
# ============================================
class Discriminator(nn.Module):
    """
    Discriminator conditioned on sketch input
    Architecture: Face + Sketch → Joint Discriminative Function → Real/Fake
    """
    def __init__(self, face_channels=3, sketch_channels=1):
        super(Discriminator, self).__init__()
        
        # Face image processing branch (x input)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(face_channels, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        
        # Sketch condition processing branch (y input)
        self.sketch_encoder = nn.Sequential(
            nn.Conv2d(sketch_channels, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        
        # Joint discriminative function (combined representation)
        # Paper uses maxout layers; we use LeakyReLU as practical alternative
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),  # 8x8 -> 4x4 (512 = 256 face + 256 sketch)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1, 4, 1, 0),  # 4x4 -> 1x1
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, face, sketch):
        """
        Forward pass: D(x|y)
        Args:
            face: Face image (real or generated) [batch, 3, H, W]
            sketch: Sketch condition [batch, 1, H, W]
        Returns:
            Probability of being real [batch]
        """
        # Process face image
        face_features = self.face_encoder(face)  # [B, 256, 8, 8]
        
        # Process sketch condition
        sketch_features = self.sketch_encoder(sketch)  # [B, 256, 8, 8]
        
        # Concatenate both features (joint input to discriminator as per paper)
        combined = torch.cat([face_features, sketch_features], dim=1)  # [B, 512, 8, 8]
        
        # Classification
        output = self.classifier(combined)  # [B, 1, 1, 1]
        return output.view(-1, 1).squeeze(1)  # [B]


# ============================================
# DATASET CLASS
# ============================================
class SketchFaceDataset(Dataset):
    """
    Dataset for paired sketch-face images
    Expected structure:
        sketch_dir/
            image1.jpg
            image2.jpg
            ...
        face_dir/
            image1.jpg
            image2.jpg
            ...
    """
    def __init__(self, sketch_dir, face_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.face_dir = face_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(sketch_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Found {len(self.image_files)} paired images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.image_files[idx])
        face_path = os.path.join(self.face_dir, self.image_files[idx])
        
        # Load images
        sketch = Image.open(sketch_path).convert('L')  # Grayscale
        face = Image.open(face_path).convert('RGB')    # RGB
        
        # Apply transforms
        if self.transform:
            sketch = self.transform(sketch)
            face = self.transform(face)
        
        return sketch, face


# ============================================
# TRAINING FUNCTION
# ============================================
def train_cgan(generator, discriminator, dataloader, num_epochs=100, 
               device='cuda', lr=0.0002, beta1=0.5, save_dir='checkpoints',
               sample_interval=10, optimizer_G=None, optimizer_D=None, start_epoch=0):
    """
    Train Conditional GAN using minimax objective (Equation 2 from paper)
    
    min_G max_D V(D,G) = E_x[log D(x|y)] + E_z[log(1 - D(G(z|y)))]
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    
    # Optimizers (Adam works well, paper uses SGD with momentum)
    if optimizer_G is None:
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    if optimizer_D is None:
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training mode
    generator.train()
    discriminator.train()
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for i, (sketches, real_faces) in enumerate(dataloader):
            batch_size = sketches.size(0)
            sketches = sketches.to(device)
            real_faces = real_faces.to(device)
            
            # Labels for real and fake data
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # =====================================
            # Train Discriminator
            # Maximize log(D(x|y)) + log(1 - D(G(z|y)))
            # =====================================
            optimizer_D.zero_grad()
            
            # Real faces with sketch condition
            real_output = discriminator(real_faces, sketches)
            d_loss_real = criterion(real_output, real_labels)
            
            # Generate fake faces conditioned on sketches
            noise = torch.randn(batch_size, 100).to(device)
            fake_faces = generator(noise, sketches)
            
            # Fake faces with sketch condition
            fake_output = discriminator(fake_faces.detach(), sketches)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # =====================================
            # Train Generator
            # Minimize log(1 - D(G(z|y))) 
            # Equivalently: maximize log(D(G(z|y)))
            # =====================================
            optimizer_G.zero_grad()
            
            # Generate fake faces again (don't reuse to avoid backprop issues)
            noise = torch.randn(batch_size, 100).to(device)
            fake_faces = generator(noise, sketches)
            
            # Try to fool discriminator
            fake_output = discriminator(fake_faces, sketches)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Print progress
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_Loss: {d_loss.item():.4f} G_Loss: {g_loss.item():.4f}')
        
        # Average losses for epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Average - D_Loss: {avg_d_loss:.4f} G_Loss: {avg_g_loss:.4f}\n')

        # Save sample images
        if (epoch + 1) % sample_interval == 0:
            save_sample_images(generator, dataloader, device, epoch + 1)

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'{save_dir}/cgan_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    print("Training complete!")


# ============================================
# UTILITY FUNCTIONS
# ============================================
def save_sample_images(generator, dataloader, device, epoch, num_samples=8):
    """Save sample generated images during training"""
    generator.eval()
    
    with torch.no_grad():
        # Get a batch of sketches
        sketches, real_faces = next(iter(dataloader))
        sketches = sketches[:num_samples].to(device)
        real_faces = real_faces[:num_samples].to(device)
        
        # Generate faces
        noise = torch.randn(num_samples, 100).to(device)
        fake_faces = generator(noise, sketches)
        
        # Denormalize from [-1, 1] to [0, 1]
        sketches = (sketches + 1) / 2
        real_faces = (real_faces + 1) / 2
        fake_faces = (fake_faces + 1) / 2
        
        # Create comparison grid
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 8))
        for i in range(num_samples):
            # Sketch
            axes[0, i].imshow(sketches[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Sketch', fontsize=12, rotation=0, labelpad=40)
            
            # Real face
            axes[1, i].imshow(real_faces[i].cpu().permute(1, 2, 0))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Real Face', fontsize=12, rotation=0, labelpad=40)
            
            # Generated face
            axes[2, i].imshow(fake_faces[i].cpu().permute(1, 2, 0))
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Generated', fontsize=12, rotation=0, labelpad=40)
        
        plt.tight_layout()
        plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    generator.train()


def weights_init(m):
    """Initialize network weights for better training stability"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================
# INFERENCE FUNCTION FOR USER INPUT
# ============================================
def generate_faces_from_sketch(generator, sketch_path, device='cuda', 
                                num_variations=5, output_dir='generated'):
    """
    Generate realistic face images from ANY user-provided sketch
    
    This is the main function for inference - users can provide any sketch
    and get multiple realistic face variations.
    
    Args:
        generator: Trained generator model
        sketch_path: Path to user's sketch image
        device: Device to run on
        num_variations: Number of face variations to generate
        output_dir: Directory to save generated faces
    
    Returns:
        List of generated face images (as PIL Images)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    generator.eval()
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Load and preprocess sketch
    print(f"Loading sketch from: {sketch_path}")
    sketch = Image.open(sketch_path).convert('L')  # Convert to grayscale
    original_sketch = sketch.copy()  # Keep original for display
    
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)  # [1, 1, 64, 64]
    
    # Generate multiple face variations
    print(f"Generating {num_variations} face variations...")
    generated_faces = []
    generated_tensors = []
    
    with torch.no_grad():
        for i in range(num_variations):
            # Sample different noise for variation
            noise = torch.randn(1, 100).to(device)
            
            # Generate face conditioned on sketch
            fake_face = generator(noise, sketch_tensor)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_face = (fake_face + 1) / 2
            
            # Convert to PIL Image
            fake_face_np = fake_face.cpu().squeeze(0).permute(1, 2, 0).numpy()
            fake_face_np = (fake_face_np * 255).astype(np.uint8)
            fake_face_pil = Image.fromarray(fake_face_np)
            
            generated_faces.append(fake_face_pil)
            generated_tensors.append(fake_face.cpu())
            
            # Save individual image
            output_path = os.path.join(output_dir, f'generated_face_{i+1}.png')
            fake_face_pil.save(output_path)
            print(f"Saved: {output_path}")
    
    # Create comprehensive visualization
    visualize_results(original_sketch, generated_tensors, output_dir)
    
    print(f"\nGeneration complete! {num_variations} faces generated.")
    print(f"Results saved to: {output_dir}/")
    
    return generated_faces


def visualize_results(sketch, generated_faces, output_dir):
    """Create a nice visualization of input sketch and generated faces"""
    num_faces = len(generated_faces)
    
    # Create figure
    fig, axes = plt.subplots(1, num_faces + 1, figsize=(3 * (num_faces + 1), 3))
    
    # Display input sketch
    axes[0].imshow(sketch, cmap='gray')
    axes[0].set_title('Input Sketch', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display generated faces
    for i, face in enumerate(generated_faces):
        axes[i+1].imshow(face.squeeze(0).permute(1, 2, 0))
        axes[i+1].set_title(f'Generated #{i+1}', fontsize=14)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_results.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_dir}/all_results.png")


# ============================================
# MAIN TRAINING SCRIPT
# ============================================
def main_train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms: separate transforms for sketch (1-channel) and face (3-channel)
    sketch_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    face_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Load dataset
    print('Loading dataset...')
    # Prefer external data_utils if available (supports root_dir or sketch/face dirs)
    try:
        from question3_cgan_sketch2face.data_utils import SketchFaceDataset as ExternalSketchFaceDataset
        print('Using SketchFaceDataset from data_utils.py')
        # try to construct using sketch/face dirs if provided, else pass root_dir
        if args.sketch_dir and args.face_dir:
            dataset = ExternalSketchFaceDataset(sketch_dir=args.sketch_dir, face_dir=args.face_dir, transform=sketch_transform, transform_face=face_transform)
        else:
            # fallback to root_dir if available
            root = getattr(args, 'root_dir', None)
            if root is None:
                # if user passed sketch_dir as a combined root, use that
                root = args.sketch_dir or args.face_dir
            dataset = ExternalSketchFaceDataset(root_dir=root, transform=sketch_transform, transform_face=face_transform)
    except Exception:
        # fallback to local implementation
        dataset = SketchFaceDataset(
            sketch_dir=args.sketch_dir,
            face_dir=args.face_dir,
            transform=sketch_transform,
            transform_face=face_transform
        )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize models
    print('Initializing models...')
    generator = Generator(
        noise_dim=args.noise_dim,
        sketch_channels=1,
        output_channels=3,
        img_size=args.img_size
    ).to(device)
    
    discriminator = Discriminator(
        face_channels=3,
        sketch_channels=1
    ).to(device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Print model info
    print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
    print(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
    
    # Prepare optimizers so we can optionally resume
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    start_epoch = 0
    # Resume if requested
    if getattr(args, 'resume_checkpoint', None):
        ckpt_path = args.resume_checkpoint
        if os.path.exists(ckpt_path):
            print(f'Loading resume checkpoint: {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device)
            # load states (try several common keys)
            if 'generator' in ckpt:
                try:
                    generator.load_state_dict(ckpt['generator'])
                    print('Loaded generator state from checkpoint')
                except Exception:
                    generator.load_state_dict(ckpt['generator'], strict=False)
            if 'discriminator' in ckpt:
                try:
                    discriminator.load_state_dict(ckpt['discriminator'])
                    print('Loaded discriminator state from checkpoint')
                except Exception:
                    discriminator.load_state_dict(ckpt['discriminator'], strict=False)
            # optimizers
            if 'optimizer_G' in ckpt:
                try:
                    optimizer_G.load_state_dict(ckpt['optimizer_G'])
                    print('Loaded optimizer_G state')
                except Exception:
                    print('Failed to fully load optimizer_G state (continuing)')
            if 'optimizer_D' in ckpt:
                try:
                    optimizer_D.load_state_dict(ckpt['optimizer_D'])
                    print('Loaded optimizer_D state')
                except Exception:
                    print('Failed to fully load optimizer_D state (continuing)')
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch'])
                print(f'Resuming from epoch {start_epoch}')
        else:
            print(f'Resume checkpoint not found: {ckpt_path} (starting from scratch)')

    # Train
    train_cgan(
        generator, 
        discriminator, 
        dataloader, 
        num_epochs=args.epochs,
        device=device,
        lr=args.lr,
        beta1=args.beta1,
        save_dir=args.save_dir,
        sample_interval=args.sample_interval,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        start_epoch=start_epoch
    )


# ============================================
# MAIN INFERENCE SCRIPT
# ============================================
def main_inference(args):
    """Main inference function - Generate faces from user sketch"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load trained generator
    print('Loading trained model...')
    generator = Generator(
        noise_dim=args.noise_dim,
        sketch_channels=1,
        output_channels=3,
        img_size=args.img_size
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Generate faces from user sketch
    generated_faces = generate_faces_from_sketch(
        generator=generator,
        sketch_path=args.sketch_path,
        device=device,
        num_variations=args.num_variations,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Successfully generated {len(generated_faces)} face images!")
    print(f"  Input sketch: {args.sketch_path}")
    print(f"  Output directory: {args.output_dir}")


# ============================================
# COMMAND LINE INTERFACE
# ============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional GAN for Sketch-to-Face Generation')
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or generate')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--sketch_dir', type=str, required=True,
                            help='Directory containing sketch images')
    train_parser.add_argument('--face_dir', type=str, required=True,
                            help='Directory containing face images')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.0002,
                            help='Learning rate')
    train_parser.add_argument('--beta1', type=float, default=0.5,
                            help='Beta1 for Adam optimizer')
    train_parser.add_argument('--noise_dim', type=int, default=100,
                            help='Dimension of noise vector')
    train_parser.add_argument('--img_size', type=int, default=64,
                            help='Image size (will be resized to img_size x img_size)')
    train_parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of data loading workers')
    train_parser.add_argument('--save_dir', type=str, default='checkpoints',
                            help='Directory to save checkpoints')
    train_parser.add_argument('--sample_interval', type=int, default=10,
                            help='Interval for saving sample images')
    train_parser.add_argument('--resume_checkpoint', type=str, default=None,
                            help='Path to checkpoint to resume training from')
    
    # Inference arguments
    gen_parser = subparsers.add_parser('generate', help='Generate faces from sketch')
    gen_parser.add_argument('--sketch_path', type=str, required=True,
                          help='Path to input sketch image')
    gen_parser.add_argument('--checkpoint_path', type=str, required=True,
                          help='Path to trained model checkpoint')
    gen_parser.add_argument('--num_variations', type=int, default=5,
                          help='Number of face variations to generate')
    gen_parser.add_argument('--output_dir', type=str, default='generated',
                          help='Directory to save generated faces')
    gen_parser.add_argument('--noise_dim', type=int, default=100,
                          help='Dimension of noise vector')
    gen_parser.add_argument('--img_size', type=int, default=64,
                          help='Image size')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_train(args)
    elif args.mode == 'generate':
        main_inference(args)
    else:
        parser.print_help()


"""
================================================================================
USAGE EXAMPLES:
================================================================================

1. TRAINING:
   python cgan_sketch_face.py train \
       --sketch_dir data/sketches \
       --face_dir data/faces \
       --epochs 100 \
       --batch_size 64

2. GENERATE FACES FROM ANY SKETCH (User Input):
   python cgan_sketch_face.py generate \
       --sketch_path my_sketch.jpg \
       --checkpoint_path checkpoints/cgan_epoch_100.pth \
       --num_variations 10

3. GENERATE WITH CUSTOM OPTIONS:
   python cgan_sketch_face.py generate \
       --sketch_path user_sketch.png \
       --checkpoint_path checkpoints/cgan_epoch_100.pth \
       --num_variations 20 \
       --output_dir my_generated_faces

================================================================================
DATASET PREPARATION:
================================================================================

Organize your Person Face Sketches dataset as:
    data/
        sketches/
            person1.jpg
            person2.jpg
            ...
        faces/
            person1.jpg  (corresponding face for person1 sketch)
            person2.jpg
            ...

The filenames must match between sketches/ and faces/ directories.

================================================================================
KEY FEATURES:
================================================================================

✓ User can provide ANY sketch image for inference
✓ Generates multiple realistic face variations from single sketch
✓ Implements conditional GAN as per original paper (Equation 2)
✓ Both generator and discriminator conditioned on sketch
✓ Saves checkpoints every 10 epochs
✓ Saves sample images during training for monitoring
✓ Easy-to-use command line interface
✓ Comprehensive visualization of results

================================================================================
"""