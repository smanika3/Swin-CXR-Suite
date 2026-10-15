"""
swin_fpn.py
Faster R-CNN detector with Swin Transformer backbone and FeaturePyramidNetwork.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import timm


class SwinBackboneWithFPN(nn.Module):
    """
    Swin Transformer backbone connected with torchvision FeaturePyramidNetwork.
    Extracts features from stages 1, 2, 3 (channels 256, 512, 1024) and projects to 256.
    """
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 512,
        out_channels: int = 256
    ):
        super().__init__()
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3],
            img_size=img_size
        )

        feature_info = self.body.feature_info.get_dicts()
        in_channels_list = [f['num_chs'] for f in feature_info]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        features = self.body(x)
        # Convert NHWC -> NCHW if necessary
        features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]

        features_dict = OrderedDict()
        for i, feat in enumerate(features):
            features_dict[str(i)] = feat

        fpn_features = self.fpn(features_dict)
        return fpn_features


def build_faster_rcnn(
    num_classes: int = 3,
    model_name: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    img_size: int = 512,
    box_score_thresh: float = 0.05
) -> FasterRCNN:
    """
    Builds a Faster R-CNN model with Swin-FPN backbone.

    Args:
        num_classes: Total classes including background (e.g., 3: Background=0, Active_TB=1, Latent_TB=2).
        model_name: Backbone model identifier.
        pretrained: Load ImageNet pretrained weights.
        img_size: Image dimension.
        box_score_thresh: Minimum classification score threshold during inference.

    Returns:
        FasterRCNN model.
    """
    backbone = SwinBackboneWithFPN(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size
    )

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=box_score_thresh,
        min_size=img_size,
        max_size=img_size
    )
    return model
