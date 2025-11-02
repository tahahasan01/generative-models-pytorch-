Question 1 - VAE and Simple GAN for Signature Generation

See the scripts in this folder. High-level:
- `data_utils.py` - dataset loader for signature images (expects local signature dataset folder).
- `models.py` - simple VAE and GAN model definitions.
- `train_vae.py` and `train_gan.py` - training scripts with CLI flags.

Run examples (PowerShell):
python train_vae.py --data_root "../data/signatures" --epochs 10
python train_gan.py --data_root "../data/signatures" --epochs 10
