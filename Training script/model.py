from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.data.transforms_factory import create_transform
from transformers import AutoConfig
from transformers import logging
logging.set_verbosity_error()
import sys

sys.path.append("/mnt/d/swinv2resumed/MambaVision-L2-1K")


from modeling_mambavision import MambaVisionModel
# ==========================================
# Weight Initialization
# ==========================================
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



# ==========================================
# 3. MMSegmentation-Equivalent FPN Head
# ==========================================
class ConvNormRelu(nn.Module):
    """Mimics MMSegmentation's ConvModule to wrap Conv2d + Norm + ReLU"""
    def __init__(self, in_c, out_c, k, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False)
        self.norm = nn.GroupNorm(32, out_c) # Change to nn.BatchNorm2d(out_c) at your own risk
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class FPNHead(nn.Module):
    """Exact replica of the FPNHead used in the Swin-TUNA codebase."""
    def __init__(self, in_channels=[192, 384, 768, 1536], channels=512, num_classes=104, dropout_ratio=0.1):
        super().__init__()
        
        # 1x1 Convs to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            ConvNormRelu(in_c, channels, k=1, p=0) for in_c in in_channels
        ])
        
        # 3x3 Convs to smooth features after top-down addition
        self.fpn_convs = nn.ModuleList([
            ConvNormRelu(channels, channels, k=3, p=1) for _ in in_channels
        ])
        
        # Final Bottleneck mapping concatenated features to head channels
        self.fpn_bottleneck = ConvNormRelu(len(in_channels) * channels, channels, k=3, p=1)
        
        self.dropout = nn.Dropout2d(p=dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        # 1. Lateral Projections
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # 2. Top-Down Fusion
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False
            )
            
        # 3. Smooth Features
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # 4. Multi-Scale Aggregation (Upsample to highest resolution feature)
        fused_shape = fpn_outs[0].shape[2:]
        fused_outs = [fpn_outs[0]]
        for i in range(1, len(fpn_outs)):
            fused_outs.append(F.interpolate(fpn_outs[i], size=fused_shape, mode='bilinear', align_corners=False))
            
        # 5. Concatenate, Bottleneck, and Classify
        x = torch.cat(fused_outs, dim=1)
        x = self.fpn_bottleneck(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        
        return x

# ==========================================
# 4. Integrated Swin-TUNA FPN Engine
# ==========================================
class MambaVisionFPN(nn.Module):
    def __init__(self, num_classes=104):
        super().__init__()

        MambaVisionModel.all_tied_weights_keys = {}


        config = AutoConfig.from_pretrained(
            "nvidia/MambaVision-L2-1K",
            trust_remote_code=True
        )

        self.backbone = MambaVisionModel.from_pretrained(
            "nvidia/MambaVision-L2-1K",
            config=config,
            trust_remote_code=True
        )

        for param in self.backbone.parameters():
            param.requires_grad = False


        # The Exact TUNA FPN Configuration
        self.decode_head = FPNHead(
            in_channels=[196, 392, 784, 1568], 
            channels=512, 
            num_classes=num_classes, 
            dropout_ratio=0.1
        )

        self.decode_head.apply(weights_init)

    
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Injection Complete. Total Trainable Parameters (TUNA + Head): {trainable:,}")

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        dd, backbone_output = self.backbone(x)
        
        features = list(backbone_output)

        logits = self.decode_head(features)
        
        # Final upsample to match input image resolution (640x640)
        output = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return output