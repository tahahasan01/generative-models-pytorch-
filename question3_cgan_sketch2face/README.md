# Conditional GAN for Sketch-to-Face Generation

Implementation of **Conditional Generative Adversarial Nets** (Mirza & Osindero, 2014) for generating realistic face images from sketches.

## 📁 Project Structure

```
cgan-sketch-to-face/
├── model.py          # Generator and Discriminator architectures
├── train.py          # Training and inference script
├── README.md         # This file
├── data/             # Your dataset (create this)
│   ├── sketches/     # Sketch images
│   └── faces/        # Corresponding face images
├── checkpoints/      # Saved models (created automatically)
├── samples/          # Training samples (created automatically)
└── generated/        # Generated faces (created automatically)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision pillow matplotlib numpy
```

### 2. Prepare Dataset

Organize your **Person Face Sketches dataset** as:

```
data/
├── sketches/
│   ├── person001.jpg
│   ├── person002.jpg
│   └── ...
└── faces/
    ├── person001.jpg  (must match sketch filename)
    ├── person002.jpg
    └── ...
```

**Important**: 
- Filenames must match between `sketches/` and `faces/`
- These are PAIRED images (same person's sketch + real face)
- Used ONLY for training

### 3. Test Model Architecture

```bash
python model.py
```

This will verify the models work correctly.

---

## 📚 Usage

### 🎓 **TRAINING** (Do this first)

Train the model on your paired sketch-face dataset:

```bash
python train.py --mode train \
    --sketch_dir data/sketches \
    --face_dir data/faces \
    --epochs 100 \
    --batch_size 64
```

**Training Options:**
```bash
--epochs 100              # Number of training epochs
--batch_size 64           # Batch size (reduce if out of memory)
--lr 0.0002              # Learning rate
--img_size 64            # Image size (64x64)
--save_dir checkpoints   # Where to save models
--sample_interval 10     # Save samples every N epochs
```

**What happens:**
- Trains for 100 epochs (may take hours/days depending on dataset size)
- Saves checkpoints every 10 epochs to `checkpoints/`
- Saves training samples every 10 epochs to `samples/`
- Prints loss values during training

**Expected output:**
```
Found 500 paired sketch-face images
Generator parameters: 3,459,843
Discriminator parameters: 2,764,801
Starting training...
Epoch [1/100] Batch [0/7] D_Loss: 1.3862 G_Loss: 0.6931
...
✓ Checkpoint saved: checkpoints/cgan_epoch_10.pth
```

---

### 🎨 **INFERENCE** (Generate from ANY sketch)

After training, generate faces from **ANY user-provided sketch**:

```bash
python train.py --mode generate \
    --sketch_path my_sketch.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth \
    --num_variations 10
```

**Inference Options:**
```bash
--sketch_path my_sketch.jpg      # YOUR sketch (can be ANY sketch!)
--checkpoint checkpoints/...     # Trained model path
--num_variations 10              # Number of faces to generate
--output_dir generated           # Output directory
```

**What happens:**
- Loads your sketch (any format: .jpg, .png, etc.)
- Generates 10 different realistic faces
- Saves results to `generated/`:
  - `generated_face_1.png`, `generated_face_2.png`, ...
  - `all_results.png` (comparison view)

**Expected output:**
```
GENERATING FACES FROM SKETCH
Input sketch: my_sketch.jpg
Number of variations: 10
Generating faces...
  ✓ Saved: generated_face_1.png
  ✓ Saved: generated_face_2.png
  ...
✓ SUCCESS! Generated 10 face variations
```

---

## 💡 Examples

### Example 1: Basic Training
```bash
python train.py --mode train \
    --sketch_dir data/sketches \
    --face_dir data/faces \
    --epochs 50 \
    --batch_size 32
```

### Example 2: Generate from Hand-Drawn Sketch
```bash
# Draw a sketch on paper, take a photo, save as my_drawing.jpg
python train.py --mode generate \
    --sketch_path my_drawing.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth \
    --num_variations 5
```

### Example 3: Generate Many Variations
```bash
python train.py --mode generate \
    --sketch_path celebrity_sketch.png \
    --checkpoint checkpoints/cgan_epoch_100.pth \
    --num_variations 20 \
    --output_dir results/celebrity
```

### Example 4: Test Different Sketches
```bash
# Generate from sketch 1
python train.py --mode generate \
    --sketch_path sketch1.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth \
    --output_dir results/sketch1

# Generate from sketch 2
python train.py --mode generate \
    --sketch_path sketch2.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth \
    --output_dir results/sketch2
```

---

## 🔍 Understanding the Files

### **model.py** - Neural Network Architectures

Contains:
- **Generator**: Takes noise + sketch → generates realistic face
  - Input: Random noise (100-dim) + Sketch image (64×64×1)
  - Output: Face image (64×64×3)
  
- **Discriminator**: Takes face + sketch → predicts real/fake
  - Input: Face image (64×64×3) + Sketch (64×64×1)
  - Output: Probability [0,1] (0=fake, 1=real)

Both networks are conditioned on the sketch (key feature of cGAN).

### **train.py** - Training & Inference

Contains:
- **Dataset loading**: Loads paired sketch-face images
- **Training loop**: Implements Equation 2 from the paper
  - Trains discriminator to distinguish real vs fake
  - Trains generator to fool discriminator
- **Inference function**: Generates faces from user sketches
- **Utilities**: Visualization, checkpointing, etc.

---

## 📊 Monitoring Training

### Check Training Progress

Look at saved samples in `samples/` directory:
```
samples/
├── epoch_10.png
├── epoch_20.png
├── epoch_30.png
└── ...
```

Each image shows:
- **Row 1**: Input sketches
- **Row 2**: Real faces (ground truth)
- **Row 3**: Generated faces

**Good training**: Generated faces should progressively look more realistic.

### Check Checkpoints

Models are saved every 10 epochs:
```
checkpoints/
├── cgan_epoch_10.pth
├── cgan_epoch_20.pth
├── cgan_epoch_30.pth
└── ...
```

You can test different checkpoints:
```bash
python train.py --mode generate \
    --sketch_path test.jpg \
    --checkpoint checkpoints/cgan_epoch_50.pth  # Try epoch 50

python train.py --mode generate \
    --sketch_path test.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth  # Try epoch 100
```

---

## 🎯 Key Features

### ✅ **For Training:**
- Uses paired sketch-face dataset
- Implements conditional GAN (Equation 2 from paper)
- Both G and D conditioned on sketch
- Automatic checkpointing
- Progress monitoring with sample images
- Loss tracking

### ✅ **For Inference:**
- Accept **ANY user sketch** (not just training data!)
- No need for paired face image
- Generate multiple variations from single sketch
- Automatic preprocessing (resize, normalize)
- Comprehensive visualization
- Easy-to-use command line

---

## 🔧 Troubleshooting

### Problem: Out of memory during training
**Solution**: Reduce batch size
```bash
python train.py --mode train ... --batch_size 16  # or even 8
```

### Problem: Training is too slow
**Solution**: Check if GPU is being used
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Generated faces look bad
**Solution**: 
1. Train for more epochs (try 150-200)
2. Check training samples to see if model is learning
3. Ensure dataset quality is good
4. Try different checkpoint (epoch 80 might be better than 100)

### Problem: "FileNotFoundError" during training
**Solution**: Check dataset structure
```bash
ls data/sketches/  # Should show image files
ls data/faces/     # Should show matching image files
```

### Problem: Sketch not found during inference
**Solution**: Use absolute path
```bash
python train.py --mode generate \
    --sketch_path /full/path/to/my_sketch.jpg \
    --checkpoint checkpoints/cgan_epoch_100.pth
```

---

## 📈 Expected Results

### After Training:
- Generator learns to create realistic faces from sketches
- Faces match sketch features (shape, pose, features)
- Each generation is slightly different (due to noise)

### Quality Depends On:
1. **Dataset size**: More data = better results (aim for 1000+ pairs)
2. **Dataset quality**: Clear, aligned sketches and faces
3. **Training epochs**: 100-200 epochs usually sufficient
4. **Model capacity**: Current architecture works for 64×64 images

---

## 🎓 How It Works (Paper Implementation)

### Conditional GAN Theory

**Standard GAN**: Generator creates random images

**Conditional GAN**: Generator creates images based on condition (sketch)

**Equation 2 (from paper)**:
```
min_G max_D V(D,G) = E_x[log D(x|y)] + E_z[log(1 - D(G(z|y)))]
```

Where:
- `x` = real face
- `y` = sketch condition
- `z` = random noise
- `G(z|y)` = generator creates face from noise + sketch
- `D(x|y)` = discriminator judges face given sketch

### Our Implementation:

1. **Generator** combines:
   - Noise vector (100-dim)
   - Sketch features (encoded)
   - Generates face that matches sketch

2. **Discriminator** checks:
   - Face image quality
   - Face-sketch consistency
   - Outputs real/fake probability

3. **Training** alternates:
   - Update D: Learn to distinguish real vs fake
   - Update G: Learn to fool D

---

## 📝 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{mirza2014conditional,
  title={Conditional generative adversarial nets},
  author={Mirza, Mehdi and Osindero, Simon},
  journal={arXiv preprint arXiv:1411.1784},
  year={2014}
}
```

---

## 🤝 Tips for Best Results

### Dataset Preparation:
1. Use high-quality sketch-face pairs
2. Ensure sketches are clear and consistent
3. Align faces (centered, similar scale)
4. Use at least 500-1000 pairs

### Training:
1. Monitor loss values (both should decrease)
2. Check sample images regularly
3. Train for enough epochs (100-200)
4. Save multiple checkpoints

### Inference:
1. Use different noise (num_variations) for variety
2. Try different checkpoints (different epochs)
3. Experiment with different sketches
4. Preprocessing is automatic, but clear sketches work best

---

## 📧 Questions?

For issues with:
- **Model architecture**: Check `model.py`
- **Training**: Check `train.py` training section
- **Inference**: Check `train.py` inference section
- **Dataset**: Verify file structure and naming

---

**Good luck with your Conditional GAN project! 🎨→🖼️**