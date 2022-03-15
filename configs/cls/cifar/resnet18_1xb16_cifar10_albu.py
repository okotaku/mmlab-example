base_dir = '/opt/conda/lib/python3.7/site-packages/mmcls/.mim/configs/'
_base_ = [
    base_dir+'_base_/models/resnet18_cifar.py',
    base_dir+'_base_/datasets/cifar10_bs16.py',
    base_dir+'_base_/schedules/cifar10_bs128.py',
    base_dir+'_base_/default_runtime.py'
]

# use albumentations
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
albu_train_transforms = [
    dict(type='ShiftScaleRotate',
         shift_limit=0.1,
         rotate_limit=20,
         scale_limit=0.1,
         p=0.5),
]
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Albu',
         transforms=albu_train_transforms,
         keymap={'img': 'image'}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
data = dict(samples_per_gpu=128,
            train=dict(pipeline=train_pipeline))
