import os
from tqdm import tqdm
import random
import shutil
import math
from datetime import datetime
# 기존의 라벨링한 이미지 제외
# 추가로 50장 구성

ROOT = 'D:/Project/globit_fish/video_data/'
TRAIN_DATA_NUM = 50


def set_testData(path, img_list , data_num):
    """_summary_

    Args:
        path (_type_): 새로 생성한 데이터의 저장 경로
        img_list (_type_): 원본 이미지 list
        data_num (_type_): 학습 데이터 갯수
    """
    
    if not os.path.isdir(path):
        os.makedirs(path + 'train')
        os.makedirs(path + 'val')

    random_img_list = random.sample(img_list, data_num)

    ratio = round(data_num * 0.1)

    for index, img in tqdm(enumerate(random_img_list)):
        if index >= ratio:
            shutil.copy(ROOT + '/images' + '/' + img,  path + 'train/' + img)
        else:
            shutil.copy(ROOT + '/images' + '/' + img,  path + 'val/' + img)


def remove_previous_data():
    previous_train_list = os.listdir(ROOT + 'testData/train/images') + os.listdir(ROOT + 'testData/val/images')

    origin_data = os.listdir(ROOT + '/images')

    for previous in previous_train_list:
        origin_data.remove(previous)
    
    return origin_data


if __name__ == "__main__": # main 함수의 선언, 시작을 의미

    img_list= remove_previous_data()

    date = datetime.today().strftime("%Y%m%d")

    set_testData(ROOT+'testData_' + date, img_list, TRAIN_DATA_NUM)
