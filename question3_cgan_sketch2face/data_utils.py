import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SketchFaceDataset(Dataset):
    def __init__(self, root_dir=None, sketch_dir=None, face_dir=None, transform=None, transform_face=None):
        """
        Flexible dataset initializer:
        - If `sketch_dir` and `face_dir` are provided, use those folders.
        - Else if `root_dir` is provided and contains `sketch`/`face` subfolders, use them.
        - Else attempt to pair images by filename under `root_dir`.
        """
        # determine directories
        if sketch_dir and face_dir:
            self.sketch_dir = sketch_dir
            self.face_dir = face_dir
        elif root_dir:
            self.sketch_dir = os.path.join(root_dir, 'sketch')
            self.face_dir = os.path.join(root_dir, 'face')
        else:
            raise ValueError('Either root_dir or both sketch_dir and face_dir must be provided')

        pairs = []
        if os.path.isdir(self.sketch_dir) and os.path.isdir(self.face_dir):
            sketch_files = sorted([f for f in os.listdir(self.sketch_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            for f in sketch_files:
                fp = os.path.join(self.face_dir, f)
                sp = os.path.join(self.sketch_dir, f)
                if os.path.exists(fp):
                    pairs.append((sp, fp))
        else:
            # search all images under root_dir and try to pair by filename prefixes
            all_files = []
            base_search_dir = root_dir or os.path.dirname(self.sketch_dir)
            for d, _, fs in os.walk(base_search_dir):
                for f in fs:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_files.append(os.path.join(d, f))
            # group by basename without extension
            byname = {}
            for p in all_files:
                name = os.path.splitext(os.path.basename(p))[0]
                byname.setdefault(name, []).append(p)
            for name, paths in byname.items():
                if len(paths) >= 2:
                    # pick first two as sketch/face
                    pairs.append((paths[0], paths[1]))

        self.pairs = pairs
        # support separate transforms for sketch and face images
        self.transform = transform or transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
        self.transform_face = transform_face or transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, f = self.pairs[idx]
        # sketch should be single-channel grayscale, face should be RGB
        s_img = Image.open(s).convert('L')
        f_img = Image.open(f).convert('RGB')
        s_t = self.transform(s_img)
        f_t = self.transform_face(f_img)
        return s_t, f_t
