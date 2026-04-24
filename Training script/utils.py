import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FoodSegDataset

def save_checkpoint(state, filename='best_model.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(dataset, batch_size, train_transform, val_transform, num_workers=12, pin_memory=True):
    train_ds = FoodSegDataset(hf_dataset_split=dataset['train'], transform=train_transform, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, persistent_workers=True, prefetch_factor=2)

    val_ds = FoodSegDataset(hf_dataset_split=dataset['validation'], transform=val_transform, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, device='cuda', num_classes=104):
    num_correct = 0
    num_pixels = 0
    total_val_loss = 0
    hist = torch.zeros((num_classes, num_classes), device=device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast('cuda'):
                preds = model(x)
                loss = loss_fn(preds, y)
                
            total_val_loss += loss.item()
            preds_cls = torch.argmax(preds, dim=1)

            num_correct += (preds_cls == y).sum()
            num_pixels += torch.numel(preds_cls)
            
            preds_flat = preds_cls.view(-1)
            y_flat = y.view(-1)
            mask = (y_flat >= 0) & (y_flat < num_classes)
            bincount_result = torch.bincount(
                num_classes * y_flat[mask].long() + preds_flat[mask].long(), 
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)
            hist += bincount_result

    pixel_acc = (num_correct / num_pixels) * 100
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    iou_per_class = intersection / (union + 1e-8)
    miou = iou_per_class.mean() * 100
    avg_val_loss = total_val_loss / len(loader)
    acc_per_class = torch.diag(hist) / (hist.sum(dim=1) + 1e-8)
    macc = acc_per_class.mean() * 100

    print(f"Val Loss: {avg_val_loss:.4f} |mAcc: {macc:.2f}% | Global Pixel Acc: {pixel_acc:.2f}% | mIoU: {miou:.2f}%")
    # model.train()
    
    return {'val_loss': avg_val_loss, 'miou': miou.item(), 'pixel_acc': pixel_acc.item(), 'per_class_iou': iou_per_class.cpu().numpy(), 'mAcc': macc.item()}

class MetricLogger:
    def __init__(self, save_dir, main_file="training_metrics_peft200.csv", class_file="per_class_iou_peft200.csv", num_classes=104):
        os.makedirs(save_dir, exist_ok=True)
        self.main_file = os.path.join(save_dir, main_file)
        self.class_file = os.path.join(save_dir, class_file)
        self.num_classes = num_classes
        
        if not os.path.isfile(self.main_file):
            with open(self.main_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_mIoU', 'Val_Pixel_Acc', 'mAcc'])
                
        if not os.path.isfile(self.class_file):
            with open(self.class_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch'] + [f'Class_{i}' for i in range(num_classes)])

    def log(self, epoch, train_loss, val_loss, miou, pixel_acc, per_class_iou=None, mAcc=None):
        with open(self.main_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, miou, pixel_acc, mAcc])
            
        with open(self.class_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch]
            if per_class_iou is not None:
                row.extend(per_class_iou.tolist())
            else:
                row.extend(['N/A'] * self.num_classes)
            writer.writerow(row)

# Include your DiceLoss, DiceCELoss, and LovaszSoftmaxLoss definitions here.


  
# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, num_classes, weight=None, dice_weight=1.0, ce_weight=1.0, ignore_index=-100):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            probas = probas[valid_mask]
            targets = targets[valid_mask]

        losses = []
        for c in range(C):
            target_c = (targets == c).float()
            if target_c.sum() == 0:
                continue

            proba_c = probas[:, c]
            errors = (target_c - proba_c).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            target_c_sorted = target_c[perm]
            grad = lovasz_grad(target_c_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        return sum(losses) / len(losses)