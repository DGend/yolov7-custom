import os
import cv2
from tqdm import tqdm

save_path = './data/labeling/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

image_list = os.listdir('./data/train/images')
label_list = os.listdir('./data/train/labels')

bbox_list = []
for label in label_list:
    file = open('./data/train/labels/'+ label, 'rt')
    lines = file.readlines()
    lines = list(map(lambda s: s.strip(), lines)) # \n 제거
    lines.insert(0, label)
    bbox_list.append(lines)

print(bbox_list[0])


label_img = []

for index, img in tqdm(enumerate(image_list)):
    temp_img = cv2.imread('./data/train/images/'+img)
    # label_img.append(temp_img)
    file_name = bbox_list[index][0].split('.')
    img_name = img.split('.')

    if file_name[0] == img_name[0]:

        for bbox in bbox_list[index][1:]:
            # print(bbox)
            points = list(map(int, list(map(float, bbox.split()))))
            print(points)
            # cv2.rectangle(temp_img, (points[1],points[2]), ((points[3], points[4])), color=(255,255,0), thickness=2)
        # print(save_path+img)
    #    cv2.imwrite(save_path+img, temp_img)










