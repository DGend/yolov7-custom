import cv2
import os
from tqdm import tqdm

def draw_bboxes_and_save(image_folder, label_folder, output_folder):
    # 이미지 폴더 내의 모든 이미지 파일을 가져옵니다.
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')
        
        # 이미지를 불러옵니다.
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        
        # 라벨 파일이 존재하는 경우, 바운딩 박스를 읽고 그립니다.
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # YOLO 포맷을 OpenCV 포맷으로 변환
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    # 바운딩 박스를 이미지에 그립니다.
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # 결과 이미지를 저장할 폴더를 확인하고, 없다면 생성합니다.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, image)
        print(f'Saved: {output_image_path}')

# 함수 사용 예시
image_folder = './data/feed_label/train/images'
label_folder = './data/feed_label/train/labels'
output_folder = './data/feed_label/train/bbox_check_images'

# 폴더가 존재하는지 확인하고 없으면 생성합니다.
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"폴더 생성: {image_folder}")
if not os.path.exists(label_folder):
    os.makedirs(label_folder)
    print(f"폴더 생성: {image_folder}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"폴더 생성: {image_folder}")

draw_bboxes_and_save(image_folder, label_folder, output_folder)
