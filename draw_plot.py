import os
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


ROOT = './plot_data/vertical_trim_All/'
label_path = ROOT + 'labels/'

short_window = 3
long_window = 7

mean_inter = True

def read_label(label_list):
    """
    YOLOv7으로 탐지 결과 정보가 있는 label.txt 파일 읽는 함수

    Args:
        label_list (_type_): YOLOv7으로 탐지한 결과 폴더인 labels의 정보 값

    Returns:
        _type_: _description_
        list : label.txt의 file 이름값
        list : 해당 label.txt에 대한 detect 갯수값
    """
    
    detect_count = []
    file_name = []
    
    for file in tqdm(label_list):
        f = open(label_path + file , 'r')
    
        f_name = file.split('_')
        file_name.append(f_name[-1])
        
        detect_count.append(len(f.readlines()))
    
    return file_name, detect_count

def clac_mean_interval(detect_count, interval = 5):
    """
    5(interval)프레임을 기준으로 탐지된 갯수 평균값 구하는 함수

    Args:
        detect_count (_type_): 해당 프레임에 탐지된 갯수
        interval (int, optional): 평균 값 구하기 위한 프레임 값(기준 값). Defaults to 5.

    Returns:
        _type_: _description_
        
        frame_interval_name : index...
        mean : 해당 프레임 기준으로 탐지된 갯수 평균 값
        
    """
    len_count = len(detect_count)

    mean = []
    
    frame_interval_name = list(range(interval - 1, len_count, interval))

    for count in frame_interval_name:
        
        sum = detect_count[count-4] + detect_count[count-3] + detect_count[count-2] + detect_count[count-1] + detect_count[count]
        mean.append(sum)
        
        sum = 0
    
    return frame_interval_name , mean


def draw_plot(x,y, save_name):
    
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y)

    plt.title('detect fish number')

    plt.ylabel('count')

    plt.savefig(ROOT + save_name)

    

def set_df(x, y):
    """
    MACD chart 구성하기 위해 dataframe 생성

    Args:
        x (_type_): index로 사용할 값
        y (_type_): 그래프로 나타날 값

    Returns:
        _type_: _description_
        
    """
    
    data = {'name': x, 'detect_count': y}
    df = pd.DataFrame(data)
    df.set_index('name', inplace=True)
    
    return df, 'detect_count'

    
def draw_macd(x, y, save_name):
    
    """
    생성된 dataframe 데이터를 사용하여 MACD 그래프 생성 및 저장
    """
    
    df , column_name = set_df(x, y)
    # 단기(Short-term) 이동평균
    
    df['Short_MA'] = df[column_name].rolling(window=short_window, min_periods=1, center=False).mean()

    # 장기(Long-term) 이동평균
    
    df['Long_MA'] = df[column_name].rolling(window=long_window, min_periods=1, center=False).mean()

    # MACD 계산
    df['MACD'] = df['Short_MA'] - df['Long_MA']

    # 시각화
    plt.figure(figsize=(10, 6))
    df[column_name].plot(label=column_name, color='blue')
    plt.legend(loc='upper left')

    # MACD 그래프 추가
    plt.twinx()
    df['MACD'].plot(label='MACD', color='red', linestyle='dashed')
    plt.legend(loc='upper right')

    plt.title('MACD Chart')

    plt.savefig(ROOT + save_name)



if __name__=="__main__":
    
    file_list = os.listdir(label_path)
    
    file_name, detect_count = read_label(file_list)
    
    draw_plot(file_name, detect_count, 'detect_fish_number.png')
    
    draw_macd(file_name, detect_count, 'detect_fish_number_MACD.png')
    
    
    if mean_inter:
    
        mean_name, detect_mean = clac_mean_interval(detect_count)
        
        draw_plot(mean_name, detect_mean, 'detect_fish_5frame_mean.png')
        
        draw_macd(mean_name, detect_mean, 'detect_fish_5frame_mean_MACD.png')
    
    
    