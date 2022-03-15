base_dir = '/opt/conda/lib/python3.7/site-packages/mmcls/.mim/configs/'
_base_ = [
    base_dir+'_base_/models/resnet18_cifar.py',
    base_dir+'_base_/datasets/cifar10_bs16.py',
    base_dir+'_base_/schedules/cifar10_bs128.py',
    base_dir+'_base_/default_runtime.py'
]

data = dict(samples_per_gpu=128)  # fix batch size
