custom_imports = dict(imports=['custom_modules.cls'],
                      allow_failed_imports=False)

# model settings
model = dict(type='ImageClassifier',
             backbone=dict(type='TIMMBackbone',
                           model_name='tf_efficientnet_b0_ns',
                           pretrained=True),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(
                 type='LinearClsHead',
                 num_classes=200,
                 in_channels=1280,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, 5),
             ))

# dataset settings
dataset_type = 'CUB'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=510),
    dict(type='RandomCrop', size=384),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=510),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = 'data/CUB_200_2011/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(type=dataset_type,
               ann_file=data_root + 'images.txt',
               image_class_labels_file=data_root + 'image_class_labels.txt',
               train_test_split_file=data_root + 'train_test_split.txt',
               data_prefix=data_root + 'images',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=data_root + 'images.txt',
             image_class_labels_file=data_root + 'image_class_labels.txt',
             train_test_split_file=data_root + 'train_test_split.txt',
             data_prefix=data_root + 'images',
             test_mode=True,
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'images.txt',
              image_class_labels_file=data_root + 'image_class_labels.txt',
              train_test_split_file=data_root + 'train_test_split.txt',
              data_prefix=data_root + 'images',
              test_mode=True,
              pipeline=test_pipeline))

evaluation = dict(
    interval=1, metric='accuracy',
    save_best='auto')  # save the checkpoint with highest accuracy

# optimizer
optimizer = dict(type='AdamW',
                 lr=1e-4,
                 weight_decay=0.0005,
                 eps=1e-8,
                 betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_iters=5,
                 warmup_ratio=0.01,
                 warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
