import numpy as np
import torch
from torch.utils.data import Dataset

class FoodSegDataset(Dataset):
    def __init__(self, hf_dataset_split, transform=None, is_train=False):
        self.hf_dataset = hf_dataset_split
        self.transform = transform
        self.is_train = is_train 

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        image = np.array(item['image'].convert("RGB"))
        mask = np.array(item['label']) 
        
        if self.transform is not None:
            if self.is_train:
                # ==================================================
                # MMSegmentation Class-Aware Crop Logic
                # ==================================================
                MAX_TRIES = 10
                for attempt in range(MAX_TRIES):
                    augmentations = self.transform(image=image, mask=mask)
                    aug_mask = augmentations['mask']
                    
                    # Calculate background ratio (0 is background in FoodSeg103)
                    bg_pixels = (aug_mask == 0).sum().item()
                    total_pixels = aug_mask.numel()
                    bg_ratio = bg_pixels / total_pixels
                    
                    # If background is 75% or less, we captured food. Break the loop.
                    if bg_ratio <= 0.75:
                        break
                        
                image = augmentations['image']
                mask = aug_mask
                
            else:
                # Standard deterministic transform for Validation/Test
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations['image']
                mask = augmentations['mask']
                
            # Convert to LongTensor for CrossEntropyLoss
            mask = mask.long() if hasattr(mask, 'long') else torch.tensor(mask, dtype=torch.long)
            
        return image, mask