base_dir = '/opt/conda/lib/python3.7/site-packages/mmdet/.mim/configs/'
_base_ = [
    base_dir + '_base_/models/faster_rcnn_r50_fpn.py',
    base_dir + '_base_/datasets/coco_detection.py',
    base_dir + '_base_/schedules/schedule_1x.py',
    base_dir + '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=16)  # fix batch size
