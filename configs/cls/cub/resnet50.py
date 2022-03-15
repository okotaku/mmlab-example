custom_imports = dict(imports=['custom_modules.cls'],
                      allow_failed_imports=False)

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
model = dict(type='ImageClassifier',
             backbone=dict(type='ResNet',
                           depth=50,
                           num_stages=4,
                           out_indices=(3, ),
                           style='pytorch',
                           init_cfg=dict(type='Pretrained',
                                         checkpoint=checkpoint,
                                         prefix='backbone')),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(
                 type='LinearClsHead',
                 num_classes=200,
                 in_channels=2048,
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
optimizer = dict(type='SGD',
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0.0005,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
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
