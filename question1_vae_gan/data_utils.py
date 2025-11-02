import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = []
        for d, _, fs in os.walk(root_dir):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(d, f))
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)
