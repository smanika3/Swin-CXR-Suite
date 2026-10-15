_base_ = [
    "mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    "coco_detection.py",
    "schedule_1x.py",
    "default_runtime.py",
]

# Custom Swin B model override
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],  # Swin B
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        drop_path_rate=0.2,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.2.0/swin_base_patch4_window7_224_22k.pth",
        ),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024])
)

classes = ("ActiveTuberculosis", "ObsoletePulmonaryTuberculosis")
data_root = "/scratch/smanika3/tbx11k-unzip/"
dataset_type = "COCODataset"
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        img_prefix=data_root + "train/img/",
        classes=classes,
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "val/img/",
        classes=classes,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "test/img/",
        classes=classes,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

train_evaluator = dict(type="CocoMetric", metric="bbox", backend_args=backend_args)
val_evaluator = dict(type="CocoMetric", metric="bbox", backend_args=backend_args)
test_evaluator = dict(type="CocoMetric", metric="bbox", backend_args=backend_args)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


# Optimizer and schedule override
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
runner = dict(type="EpochBasedRunner", max_epochs=25)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
work_dir = "/scratch/smanika3/checkpoints-local/tbx_swin_exp1"
