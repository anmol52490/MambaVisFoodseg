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
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.Dropout2d(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))
        self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
            nn.Dropout2d(p=0.30),
            nn.PReLU(num_parameters=1, init=0.25))

    def forward(self, x):
        out1 = self.layer1(x)
        out1 = self.layer3(out1)
        return out1



class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)

class PSPhead(nn.Module):
    def __init__(self, input_dim=1568, output_dims=392, final_output_dims=1568, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool),
                nn.Conv2d(input_dim, output_dims, kernel_size=1),
                nn.BatchNorm2d(output_dims),
                nn.PReLU(num_parameters=1, init=0.25)
            )
            for pool in pool_scales
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x):
        ppm_outs = [x]
        for ppm in self.ppm_modules:
            ppm_out = F.interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        
        ppm_outs = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(ppm_outs)
        return x
    

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[192, 384, 768, 1536], fpn_out=192):
        super(FPN_fuse, self).__init__()
        
        # 1. Lateral convolutions applied to ALL stages to ensure feature adaptation
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_out, kernel_size=1)
            for in_ch in feature_channels
        ])

        # 2. Smoothing convolutions for aliasing reduction
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
            for _ in range(len(feature_channels))
        ])

        # 3. Final Fusion bottlenecks concatenated FPN maps back to standard depth
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # Step 1: Uniform lateral projection
        lats = [lateral(f) for lateral, f in zip(self.lateral_convs, features)]

        # Step 2: Strict Top-Down Summation
        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i-1].shape[2:], mode='bilinear', align_corners=True)
            lats[i-1] = lats[i-1] + up

        # Step 3: Anti-aliasing smoothing
        ps = [smooth(l) for smooth, l in zip(self.smooth_convs, lats)]

        # Step 4: Multi-Scale Aggregation (Targeting Stage 1 resolution)
        target_h, target_w = ps[0].shape[2:]
        fused_ps = [ps[0]]
        for i in range(1, len(ps)):
            fused_ps.append(F.interpolate(ps[i], size=(target_h, target_w), mode='bilinear', align_corners=True))

        # Output Channel Flow: 4 * 192 = 768 -> conv_fusion -> 192
        x = torch.cat(fused_ps, dim=1)
        return self.conv_fusion(x)

        
# ==========================================
# 4. Integrated Swin-TUNA FPN Engine
# ==========================================
class MambaVisionUperNet(nn.Module):
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
        self.feature_channels = [196, 392, 784, 1568]

        self.PPMhead = PSPhead(input_dim=1568, output_dims=392, final_output_dims=1568)
        self.FPN = FPN_fuse(self.feature_channels, fpn_out=512)
        
        # Head specifically expects the 192 output from the corrected FPN_fuse
        self.head = ConvLayer(512, 128, kernel_size=3, stride=1, padding=1, output=64, layertype=3, droupout=True)
        self.ClassifyBlock = ClassifyBlock(64, num_classes)

    
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Injection Complete. Total Trainable Parameters (TUNA + Head): {trainable:,}")

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        dd, backbone_output = self.backbone(x)
        
        features = list(backbone_output)
        

        features[-1] = self.PPMhead(features[-1])
        
        x = self.FPN(features)
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        x = self.ClassifyBlock(x)

        return x