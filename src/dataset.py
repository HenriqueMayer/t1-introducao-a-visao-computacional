import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import sys

# Ensure src path is available
sys.path.append(str(Path(__file__).parent.resolve()))
import noise_filter

class OilSpillDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, filter_type: str = "none", img_size: int = 256):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.filter_type = filter_type
        self.img_size = img_size
        
        self.img_paths = sorted(list(self.img_dir.glob("*.png")))
        self.valid_img_paths = []
        for p in self.img_paths:
            if (self.mask_dir / p.name).exists():
                self.valid_img_paths.append(p)
                
    def __len__(self):
        return len(self.valid_img_paths)
        
    def __getitem__(self, idx):
        img_path = self.valid_img_paths[idx]
        mask_path = self.mask_dir / img_path.name
        
        img = noise_filter.load_image(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)
            
        img_filtered = noise_filter.filter_img(img, self.filter_type)
        
        if img_filtered.shape != (self.img_size, self.img_size):
            img_filtered = cv2.resize(img_filtered, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape != (self.img_size, self.img_size):
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
        img_norm = img_filtered.astype(np.float32) / 255.0
        mask_norm = (mask > 0).astype(np.float32)
        
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_norm, dtype=torch.float32).unsqueeze(0)
        
        return img_tensor, mask_tensor
