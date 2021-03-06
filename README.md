# MMLabExample

## About OpenMMLab

- [official web site](https://openmmlab.com/)
- [github](https://github.com/open-mmlab)
- [platform](https://platform.openmmlab.com/home/)
- [twitter](https://twitter.com/OpenMMLab)
- [zhihu](https://www.zhihu.com/people/openmmlab)
- [CVPR 2021 Tutorial](https://openmmlab.com/community/cvpr2021-tutorial)

## Download Dataset

COCO can be downloaded from [COCO homepage](https://cocodataset.org/#home).
Also mmdet has [a download script](https://github.com/open-mmlab/mmdetection/blob/master/tools/misc/download_dataset.py).

CUB can be downloaded from [CUB homepage](http://www.vision.caltech.edu/datasets/cub_200_2011/).

```
coco
├── annotations
├── train2017
└── val2017

cub
└── CUB_200_2011
    ├── image_class_labels.txt
    ├── images
    ├── images.txt
    └── train_test_split.txt
```

## Environment setup

```commandline
export CUB_DIR="/path/to/cub"
export COCO_DIR="/path/to/coco"
```

## How to Run

### Cifar10

```
docker-compose up -d mmcls_example

# Train
docker-compose exec mmcls_example mim train mmcls configs/cls/cifar/resnet18_1xb16_cifar10.py --gpus 1
# Test
docker-compose exec mmcls_example mim test mmcls configs/cls/cifar/resnet18_1xb16_cifar10.py --gpus 1 --checkpoint work_dirs/resnet18_1xb16_cifar10/epoch_200.pth --metrics accuracy

docker-compose stop mmcls_example
docker-compose rm mmcls_example
```

### CUB

```
docker-compose up -d mmcls_example

# Train
docker-compose exec mmcls_example mim train mmcls configs/cls/cub/resnet50.py --gpus 1
# Test
docker-compose exec mmcls_example mim test mmcls configs/cls/cub/resnet50.py --gpus 1 --checkpoint work_dirs/resnet50/epoch_100.pth --metrics accuracy

docker-compose stop mmcls_example
docker-compose rm mmcls_example
```

### COCO
```
docker-compose up -d mmdet_example

# Train
docker-compose exec mmdet_example mim train mmdet configs/det/faster_rcnn_r50_fpn_1x_coco.py --gpus 2 --launcher pytorch
# Test
docker-compose exec mmdet_example mim test mmdet configs/det/faster_rcnn_r50_fpn_1x_coco.py --gpus 1 --checkpoint work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth --eval bbox

docker-compose stop mmdet_example
docker-compose rm mmdet_example
```
