"""
ProtoGCN Config for Fall Detection - Joint-Motion Modality
"""

modality = 'jm'  # Joint-Motion
graph = 'coco17'
work_dir = f'./work_dirs/fall_detection/jm'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        num_prototype=100,  # Reduced from 400 (smaller dataset)
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(
            layout=graph,
            mode='random',
            num_filter=8,
            init_off=.04,
            init_std=.02
        )
    ),
    cls_head=dict(
        type='SimpleHead',
        joint_cfg=graph,  # Use custom COCO graph
        num_classes=2,  # Fall/No Fall
        in_channels=384,
        weight=0.2
    )
)

dataset_type = 'PoseDataset'
ann_file = 'data/fall_detection/fall_detection.pkl'

train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=30),  # 30 frames for Fall
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=30, num_clips=1),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=30, num_clips=10),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,  # Batch size per GPU
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='xsub_train'
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val'
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_test'  # Use test split instead of val
    )
)

# Optimizer (adjusted for smaller dataset)
# NTU: 4 GPU × 16 batch = 64 total, LR = 0.1
# Fall: 1 GPU × 16 batch = 16 total, LR = 0.025 (scale down)
optimizer = dict(
    type='SGD',
    lr=0.025,  # Lower LR for smaller dataset
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

total_epochs = 100  # Fewer epochs than NTU (150)

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

# Runtime settings
dist_params =dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
