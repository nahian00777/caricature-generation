# utils/datasets.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CaricaturePairsDataset(Dataset):
    def __init__(self, pairs_csv, transform=None):
        import csv
        self.items=[]
        with open(pairs_csv) as f:
            reader=csv.DictReader(f)
            for r in reader:
                self.items.append((r['photo'], r['caricature']))
        self.root = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        self.transform = transform or T.Compose([T.Resize((256,256)), T.ToTensor()])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ppath, cpath = self.items[idx]
        p = Image.open(os.path.join(self.root, ppath)).convert('RGB')
        c = Image.open(os.path.join(self.root, cpath)).convert('RGB')
        return self.transform(p), self.transform(c)
