# util/fmcw_test_util.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import glob

class FMCWDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, num_classes=5):
        self.root_dir = root_dir
        self.num_points = num_points
        self.num_classes = num_classes

        # Finde alle .ply-Dateien in allen Unterordnern
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.ply"), recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f"Keine .ply-Dateien in {root_dir} gefunden")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        pcd = o3d.io.read_point_cloud(filepath)
        xyz = np.asarray(pcd.points)

        if xyz.shape[0] == 0:
            xyz = np.zeros((1, 3))

        if xyz.shape[0] >= self.num_points:
            indices = np.random.choice(xyz.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(xyz.shape[0], self.num_points, replace=True)

        points = xyz[indices]
        label = np.random.randint(0, self.num_classes)

        return torch.tensor(points, dtype=torch.float32), label
