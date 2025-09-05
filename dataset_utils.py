import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images, scribbles, ground_truth=None, transform=None):
        self.images = images
        self.scribbles = scribbles
        self.ground_truth = ground_truth
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        scribble = self.scribbles[idx]
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Create input with scribbles as additional channel
        # Scribble: 0=background, 1=foreground, 255=unlabeled -> convert to -1, 1, 0
        scribble_input = np.where(scribble == 255, 0, np.where(scribble == 0, -1, 1))
        scribble_input = torch.from_numpy(scribble_input).unsqueeze(0).float()
        
        # Concatenate image and scribble
        input_tensor = torch.cat([image, scribble_input], dim=0)  # 4 channels: RGB + scribble
        
        if self.ground_truth is not None:
            gt = torch.from_numpy(self.ground_truth[idx]).long()
            return input_tensor, gt
        
        return input_tensor

def create_data_loader(images, scribbles, ground_truth=None, batch_size=4, shuffle=True):
    dataset = SegmentationDataset(images, scribbles, ground_truth)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)