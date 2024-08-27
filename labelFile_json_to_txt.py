"""
shell 명령어
python labelFile_json_to_txt.py --json_folder ./data/feed_label/train/json_labels --output_folder ./data/feed_label/train/labels

"""


import json
import os
from tqdm import tqdm
import argparse

def convert_json_to_yolo(json_folder, output_folder, img_width=3840, img_height=2160):
    # 주어진 폴더에서 모든 .json 파일을 찾습니다.
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_folder, json_file)
        output_txt_path = os.path.join(output_folder, os.path.splitext(json_file)[0] + '.txt')
        
        # JSON 파일을 열고 데이터 로드
        with open(json_path, 'r') as file:
            data = json.load(file)

        annotations = data['annotations']
        
        # 변환 결과를 저장할 문자열 리스트
        yolo_format_lines = []

        # 각 어노테이션을 YOLO 포맷으로 변환
        for ann in annotations:
            # 바운딩 박스 좌표를 추출
            x_min, y_min, x_max, y_max = ann['bbox']

            # YOLO 포맷으로 중심 좌표와 너비/높이 계산
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 클래스 ID (여기서는 0부터 시작하는 클래스 인덱스 사용, 'fish' 클래스는 인덱스 0)
            class_id = 0

            # 변환된 데이터를 문자열로 포맷팅하여 리스트에 추가
            line = f"{class_id} {x_center} {y_center} {width} {height}"
            yolo_format_lines.append(line)

        # 모든 어노테이션을 변환한 후 결과를 텍스트 파일로 저장
        with open(output_txt_path, 'w') as file:
            for line in yolo_format_lines:
                file.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON labels to YOLO format')
    parser.add_argument('--json_folder', type=str, default='./data/feed_label/val/json_labels', help='Path to the folder containing JSON files')
    parser.add_argument('--output_folder', type=str, default='./data/feed_label/val/labels', help='Path to the folder to save the converted labels')
    args = parser.parse_args()

    convert_json_to_yolo(args.json_folder, args.output_folder)
    
    # JSON 파일들이 있는 폴더와 결과를 저장할 폴더 경로 설정

    # 함수 호출
    convert_json_to_yolo(args.json_folder, args.output_folder)