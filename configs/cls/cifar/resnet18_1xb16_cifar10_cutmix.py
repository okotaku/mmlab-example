base_dir = '/opt/conda/lib/python3.7/site-packages/mmcls/.mim/configs/'
_base_ = [
    base_dir+'_base_/models/resnet18_cifar.py',
    base_dir+'_base_/datasets/cifar10_bs16.py',
    base_dir+'_base_/schedules/cifar10_bs128.py',
    base_dir+'_base_/default_runtime.py'
]

# use cutmix
model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)),
    train_cfg=dict(
        augments=dict(type='BatchCutMix', alpha=1.0, num_classes=10,
                      prob=1.0))
)

data = dict(samples_per_gpu=128)  # fix batch size
