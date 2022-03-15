# MMLabExample

## Download COCO

It can be downloaded from [COCO homepage](https://cocodataset.org/#home).
Also mmdet has [a download script](https://github.com/open-mmlab/mmdetection/blob/master/tools/misc/download_dataset.py).

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
