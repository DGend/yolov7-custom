# Finsh Autolabeling 프로그램
해당 프로그램은 YOLOv7을 사용하여 어류 BBox 라벨링 작업을 돕는 프로그램입니다.

# YOLOv7이란?

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

## Training

Custom training 수행방법

Step 1. 학습 이미지 저장 경로('yolov7-custom/data/feed_label/train/images')에 신규 학습용 이미지 데이터 복사

Step 2. 학습 라벨링 저장 경로('yolov7-custom/data/feed_label/train/labels')에 신규 학습용 이미지 데이터의 라벨 파일을 복사

Step 3. 다음 명령어 실행

``` shell
python train.py --workers 1 --device 0 --batch-size 32 --epochs 300 --img 640 640 --data data/fish.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-fish.yaml --weights yolov7.pt --name yolov7-halibut
```

## Testing

Custom test 수행방법
Step 1. 검증 이미지 저장 경로('yolov7-custom/data/feed_label/val/images')에 학습용 이미지로 사용되지 않은 데이터를 복사

Step 2. 검증 라벨링 저장 경로('yolov7-custom/data/feed_label/val/labels')에 학습용 이미지로 사용되지 않은 데이터를 복사

Step 3. 다음 명령어 실행

``` shell
# 파일 단위 실행
python detect.py --weights yolov7_fish.pt --conf 0.5 --img-size 640 --source yolov7-custom/data/feed_label/val/1.jpg --view-img --no-trace

# 폴더 단위 실행
python detect.py --weights yolov7_fish.pt --conf 0.5 --img-size 640 --source yolov7-custom/data/feed_label/val --view-img --no-trace
```


## Label변환

글로비트에서 제공하는 라벨은 .json으로 학습을 위해서는 형태를 변경해야한다. 따라서 다음 명령어를 통해 학습이 가능한 .txt로 변환

``` shell
python labelFile_json_to_txt.py