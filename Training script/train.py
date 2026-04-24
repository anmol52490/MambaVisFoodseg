import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# Enable TF32 for extreme speedups on Ampere GPUs (A5000)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import time
import datetime
import cv2

from model import MambaVisionFPN
from utils import get_loaders, check_accuracy, save_checkpoint, MetricLogger, DiceCELoss, LovaszSoftmaxLoss

# --- Hyperparameters ---
LR = 1e-4 # Higher LR because we are training from scratch (Adapters + Decoder)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8# SwinV2 + UperNet uses heavy VRAM; reduced to 16.
TOTAL_EPOCHS = 200
EVAL_FREQ = 5
LOSS_SWITCH_EPOCH = int(TOTAL_EPOCHS * 0.85)
IMG_HEIGHT = 640
IMG_WIDTH = 640
IMG_SIZE = 640
ACCUM_STEPS = 2

def train_fn(loader, model, optimizer, loss_fn, accum_steps):
    model.train()
    loop = tqdm(loader, leave=False, file=sys.stdout, dynamic_ncols=True)

    total_loss = 0.0
    batch_losses = []

    # Clear any residual gradients before the epoch begins
    optimizer.zero_grad(set_to_none=True)

    for step, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, non_blocking=True)
        targets = targets.long().to(device=DEVICE, non_blocking=True)

        # Execute in BF16 to halve activation VRAM natively
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            # Scale the loss to average the accumulated gradients
            loss = loss / accum_steps

        # Accumulate gradients into the .grad buffers
        loss.backward()

        # Execute weight update only when accumulation target is hit, 
        # or if it is the absolute last batch of the dataset.
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Restore the loss to its true scale for accurate CSV logging
        loss_val = loss.item() * accum_steps 
        total_loss += loss_val
        batch_losses.append(loss_val)
        loop.set_postfix(loss=loss_val)

    return total_loss / len(loader), batch_losses

def main():
    script_start_time = time.time()
    
    train_transform = A.Compose([
    A.SmallestMaxSize(max_size=IMG_SIZE, p=1.0),
    A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
    A.PadIfNeeded(
        min_height=IMG_SIZE, 
        min_width=IMG_SIZE, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=[0, 0, 0], 
        fill_mask=0 
    ),
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


    val_transform = A.Compose([
    A.SmallestMaxSize(max_size=IMG_SIZE, p=1.0),
    A.PadIfNeeded(
        min_height=IMG_SIZE, 
        min_width=IMG_SIZE, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=[0, 0, 0], 
        fill_mask=0
    ),
    A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely creates the save path inside that directory
    save_dir = os.path.join(BASE_DIR, "Output", "epochs_200_Mvi_FPN_640")
    os.makedirs(save_dir, exist_ok=True)

    batch_loss_file = os.path.join(save_dir, "batch_losses_peft.csv")
    if not os.path.isfile(batch_loss_file):
        with open(batch_loss_file, mode='w', newline='') as f:
            f.write("Epoch,Batch_Index,Loss\n")

    # Initialize Model and Compile
    model = MambaVisionFPN(num_classes=104).to(DEVICE)
    # print("=> Compiling Model with torch.compile...")
    # model = torch.compile(model) # Compiles the execution graph for speed
    # model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # print(model.backbone.is_gradient_checkpointing)

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Place this right after initializing your model
    trainable_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total trainable params: {trainable_params1:,}")
    print(f"Total params: {total_params:,}")
    print(f"{(trainable_params1 / total_params) * 100:.2f}% of params are trainable")
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    # scaler = torch.amp.GradScaler('cuda')

    weights_path = "/mnt/d/swinv2resumed/Swinv2UpernetFoodseg/class_weights.pt"
    if os.path.exists(weights_path):
        print("=> Loading smoothed Inverse Frequency Class Weights...")
        class_weights = torch.load(weights_path).to(DEVICE)
    else:
        print("=> WARNING: class_weights.pt not found. Using uniform weights.")
        class_weights = None
    
    dice_ce_loss = DiceCELoss(num_classes=104, weight=class_weights)
    lovasz_loss = LovaszSoftmaxLoss()
    active_loss_fn = dice_ce_loss

    dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir="/mnt/d/swinv2resumed/FoodSegWithUnet/data/")
    train_loader, val_loader = get_loaders(dataset, BATCH_SIZE, train_transform, val_transform)

    logger = MetricLogger(save_dir= save_dir,main_file="metrics_peft200.csv", class_file="iou_peft200.csv")
    best_miou = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)


    start_epoch = 1
    resume_path = os.path.join(save_dir, "checkpoints", "latest_training_state.pth.tar")
    
    if os.path.isfile(resume_path):
        print(f"=> Loading checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        
        model.load_state_dict(checkpoint['state_dict'])

        print("=> Pre-allocating contiguous VRAM blocks to prevent fragmentation...")
        model.train()


        dummy_data = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        dummy_data.requires_grad_(True) # Forces PyTorch to build the full memory graph

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            dummy_preds = model(dummy_data)
            dummy_loss = dummy_preds.sum()
        dummy_loss.backward()

        optimizer.zero_grad(set_to_none=True)
        del dummy_data, dummy_preds, dummy_loss

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1  # Start at the next epoch

        # 3. Destroy the 1.5GB dictionary object immediately
        del checkpoint
        
        # 4. Force Python garbage collection and wipe the CUDA memory cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"=> Successfully resumed from epoch {start_epoch - 1}. Next up: Epoch {start_epoch}")
    else:
        print("=> No resume checkpoint found. Starting from scratch.")
    
    print("--- Starting Training ---")
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{TOTAL_EPOCHS}]")
        
        if epoch == LOSS_SWITCH_EPOCH:
            print("=> Phase 2: Switching to LovaszSoftmaxLoss.")
            active_loss_fn = lovasz_loss

        avg_train_loss, current_batch_losses = train_fn(train_loader, model, optimizer, active_loss_fn, ACCUM_STEPS)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        with open(batch_loss_file, mode='a', newline='') as f:
            for b_idx, b_loss in enumerate(current_batch_losses):
                f.write(f"{epoch},{b_idx},{b_loss}\n")

        scheduler.step()

        if epoch % EVAL_FREQ == 0 or epoch >= LOSS_SWITCH_EPOCH:
            print("=> Evaluating...")
            metrics = check_accuracy(val_loader, model, active_loss_fn, device=DEVICE)
            
            logger.log(epoch, avg_train_loss, metrics['val_loss'], metrics['miou'], metrics['pixel_acc'], metrics['per_class_iou'], metrics['mAcc'])

            if metrics['miou'] > best_miou:
                best_miou = metrics['miou']
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                # save_dir = "Output/epochs_200_Mvi_FPN_640"
                model_dir = os.path.join(save_dir, "models")

                os.makedirs(model_dir, exist_ok=True)

                filename = os.path.join(
                    model_dir,
                    f"{best_miou:.2f}MIOU_{avg_train_loss:.2f}Loss_{metrics['pixel_acc']:.2f}pixAcc_{metrics['mAcc']:.2f}mAcc_model.pth.tar"
                    )
                save_checkpoint(checkpoint, filename=filename)
        else:
            logger.log(epoch, avg_train_loss, "N/A", "N/A", "N/A", None, None)

        latest_checkpoint = {
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': avg_train_loss
        }

        # save_dir = "Output/epochs_200_Mvi_FPN_640"
        chkpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)
        
        # Static name to force OVERWRITE and save hard drive space.
        # The exact epoch and loss are safely stored inside the dictionary above.
        latest_filename = os.path.join(chkpt_dir, "latest_training_state.pth.tar")
        save_checkpoint(latest_checkpoint, filename=latest_filename)


    torch.cuda.synchronize(device=DEVICE)
    script_end_time = time.time()
    formatted_time = str(datetime.timedelta(seconds=int(script_end_time - script_start_time)))
    print("\n" + "="*50)
    print(f"✅ SCRIPT COMPLETE: Total execution time was {formatted_time}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()