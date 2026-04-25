import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from model import MambaVisionFPN
from utils import load_checkpoint
import random



# Standard FoodSeg103 Color Palette generator logic
np.random.seed(42)
FOODSEG_PALETTE = np.random.randint(0, 255, size=(104, 3), dtype=np.uint8)
FOODSEG_PALETTE[0] = [0, 0, 0] # Set background to black

def decode_segmentation_mask(mask, palette):
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(palette)):
        rgb_image[mask == class_id] = palette[class_id]
    return rgb_image
# ... [Keep your FOODSEG_PALETTE and decode_segmentation_mask functions unchanged] ...

def visualize_prediction(model, dataset_split, image_index, val_transform, device="cuda"):
    model.eval()
    item = dataset_split[image_index]
    raw_image = np.array(item['image'].convert("RGB"))
    raw_mask = np.array(item['label'])
    
    augmentations = val_transform(image=raw_image, mask=raw_mask)
    input_tensor = augmentations['image'].unsqueeze(0).to(device)
    true_mask = augmentations['mask'].numpy()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
    true_mask_rgb = decode_segmentation_mask(true_mask, FOODSEG_PALETTE)
    pred_mask_rgb = decode_segmentation_mask(pred_mask, FOODSEG_PALETTE)
    
    # Un-normalize for visualization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    input_img_vis = input_tensor.squeeze(0).cpu().numpy() * std + mean
    input_img_vis = np.clip(input_img_vis.transpose(1, 2, 0), 0, 1) 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(input_img_vis); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(true_mask_rgb); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
    axes[2].imshow(pred_mask_rgb); axes[2].set_title("Prediction"); axes[2].axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint_path = r"epochs_50_224_manual_arch/models/39.62MIOU_0.34Loss_79.68pixAcc_51.79mAcc_model.pth.tar"
    checkpoint_path= r"/mnt/d/swinv2resumed/MambaVisFoodseg/Training script/Output/epochs_200_Mvi_FPN_640/models/35.51MIOU_0.39Loss_73.60pixAcc_48.81mAcc_model.pth.tar"
    
    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    inference_model = MambaVisionFPN(num_classes=104).to(device)
    load_checkpoint(torch.load(checkpoint_path, map_location=device), inference_model)
    # ckpt = torch.load(checkpoint_path, map_location=device)
    # inference_model.load_state_dict(ckpt)
    

    dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir="/mnt/d/swinv2resumed/FoodSegWithUnet/data/")

    idx = random.randint(0, len(dataset['validation']) - 1)
    print(f"Visualizing validation image #{idx}...")
    visualize_prediction(inference_model, dataset['validation'], idx, val_transform, device)