"""
swin_fpn.py
Modular Faster R-CNN detector with a Swin Transformer backbone and Feature Pyramid Network (FPN).
"""

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import timm


class SwinBackboneFPN(nn.Module):
    """
    Swin Transformer backbone connected to an FPN projection.
    Extracts multi-scale hierarchical feature representations from 4 stages.
    """
    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        img_size: int = 800,
        out_channels: int = 256
    ):
        super().__init__()
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size
        )

        # Swin-B default channels across stages: [128, 256, 512, 1024]
        # Inspect dynamically from timm feature info if available
        feature_info = getattr(self.swin, 'feature_info', None)
        if feature_info is not None:
            swin_channels = [info['num_chs'] for info in feature_info]
        else:
            swin_channels = [128, 256, 512, 1024]

        self.fpn_proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in swin_channels
        ])
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        feats = self.swin(x)
        out = {}
        for i, (feat, proj) in enumerate(zip(feats, self.fpn_proj)):
            # Swin output is [B, H, W, C] in some timm versions; permute to [B, C, H, W] if needed
            if feat.dim() == 4 and feat.shape[-1] == proj.in_channels and feat.shape[1] != proj.in_channels:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            feat = proj(feat)
            out[str(i)] = feat
        return out


def make_fasterrcnn_swin(
    num_classes: int = 2,
    model_name: str = 'swin_base_patch4_window7_224',
    pretrained: bool = True,
    img_size: int = 800,
    fpn_channels: int = 256
) -> FasterRCNN:
    """
    Constructs a Faster R-CNN model with Swin Transformer FPN backbone.

    Args:
        num_classes: Number of foreground classes (background is implicitly class 0 in torchvision).
        model_name: timm model identifier for Swin Transformer.
        pretrained: Whether to initialize with ImageNet pre-trained weights.
        img_size: Expected input image resolution.
        fpn_channels: Number of output channels for the FPN layers.

    Returns:
        torchvision.models.detection.FasterRCNN instance.
    """
    backbone = SwinBackboneFPN(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        out_channels=fpn_channels
    )

    anchor_gen = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    # Note: torchvision FasterRCNN requires num_classes to include the background class
    # For 2 TB classes (Active TB, Obsolete TB), total classes = 2 + 1 = 3
    total_classes = num_classes + 1

    model = FasterRCNN(
        backbone=backbone,
        num_classes=total_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pooler
    )
    return model
