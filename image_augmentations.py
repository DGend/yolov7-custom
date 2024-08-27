import imageio
import os

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
from tqdm import tqdm

def load_dataset(image_dir, label_dir):
    images = []
    bbs = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

            # 이미지 로드
            image = imageio.imread(image_path)
            images.append(image)

            # 라벨 로드 및 바운딩 박스 객체 생성
            bounding_boxes = []
            with open(label_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            bbs.append(BoundingBoxesOnImage(bounding_boxes, shape=image.shape))

    return images, bbs

def augment_images(images, bbs):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.Flipud(0.5),  # 상하 반전
        iaa.Affine(rotate=(-60, 60)),  # -25도에서 25도 사이로 회전
    ])

    # 이미지와 바운딩 박스 증강
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    return images_aug, bbs_aug

# 데이터셋 로드
image_dir = './data/feed_label/train/images' # 'path/to/train/images'
label_dir = './data/feed_label/train/labels' # 'path/to/train/labels'

save_image_dir = './data/feed_label/train/imgaug/images'
save_label_dir = './data/feed_label/train/imgaug/labels'
images, bbs = load_dataset(image_dir, label_dir)

# 이미지 증강
images_aug, bbs_aug = augment_images(images, bbs)

# 결과 저장
for idx, (image_aug, bb_aug) in enumerate(tqdm(zip(images_aug, bbs_aug), total=len(images_aug))):
    imageio.imwrite(f'{save_image_dir}/image_{idx}.jpg', image_aug)
    with open(f'{save_label_dir}/label_{idx}.txt', 'w') as f:
        for bb in bb_aug.bounding_boxes:
            f.write(f'{bb.x1} {bb.y1} {bb.x2} {bb.y2}\n')
