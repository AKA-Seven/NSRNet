from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img
   

    def __len__(self):
        return len(self.samples)


class ImageFolder_coco(Dataset):
    def __init__(self, root, root_sec, transform=None, split="train"):
        splitdir = Path(root) / split
        splitdir_sec = Path(root_sec) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        if not splitdir_sec.is_dir():
            raise RuntimeError(f'Invalid directory "{root_sec}"')

        
        
        
        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.samples_sec = sorted([f for f in splitdir_sec.iterdir() if f.is_file()])
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        img_sec = Image.open(self.samples_sec[index]).convert("RGB")
        if self.transform:
            return self.transform(img), self.transform(img_sec)
        return img
   

    def __len__(self):
        return len(self.samples)