import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = './plot_data/'

FOLDER_NAME  = 'vertical_trim_All/'

def read_file():
    """
    YOLOv7으로 탐지 결과 정보가 있는 label.txt 파일 읽는 함수

    Returns:
        _type_: _description_
        
        list : 해당 label.txt에 대한 detect 갯수값
    """
    
    path = ROOT + FOLDER_NAME + 'labels/'
    file_list = os.listdir(path)
    
    detect_count = []
    
    for file in tqdm(file_list):
        f = open(path + file , 'r')
        detect_count.append(len(f.readlines()))
    
    return detect_count


def moving_average(data, window_size):
    """
    이동 평균 적용

    Args:
        data (_type_): 프레임 별 YOLOv7 탐지된 갯수 = detect_count
        window_size (_type_): 이동 평균을 적용할 프레임 수

    Returns:
        _type_: _description_
        
        list : wiindow_size 값에 따라 이동 평균을 적용한 최종 값
    """
    
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def exponential_smoothing(data, alpha):
    """
    지수평활법 적용

    Args:
        data (_type_): 프레임 별 YOLOv7 탐지된 갯수 = detect_count
        alpha (_type_):  0 < α < 1 범위로 현재 관측치에 대한 가중치를 조절하는 값

    Returns:
        _type_: _description_
        list : alpha 값에 따라 지수평활법을 적용한 최종 값
    """
    
    result = [data[0]]  # 초기 예측값은 첫 번째 데이터 포인트
    for t in range(1, len(data)):
        result.append(alpha * data[t] + (1 - alpha) * result[t - 1])
    return result

def double_exponential_smoothing(data, alpha, beta):
    """
    이중 지수 평활법 적용

    Args:
        data (_type_): 프레임 별 YOLOv7 탐지된 갯수 = detect_count
        alpha (_type_): 0 < α < 1 범위로 현재 관측치에 대한 가중치  
        beta (_type_):  0 < β < 1 범위로 추세에 대한 가중치

    Returns:
        _type_: alpha, beta 값에 따라 지수평활법을 적용한 최종 값
    """
    level, trend = data[0], data[1] - data[0]
    result = [data[0]]
    for t in range(1, len(data)):
        level = alpha * data[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - result[t - 1]) + (1 - beta) * trend
        result.append(level + trend)
    return result

def triple_exponential_smoothing(data, alpha, beta, gamma, seasonality_period):
    """
    삼중 지수 평활법 적용

    Args:
        data (_type_): 프레임 별 YOLOv7 탐지된 갯수 = detect_count
        alpha (_type_): 0 < α < 1 범위로 현재 관측치에 대한 가중치
        beta (_type_): 추세에 대한 가중치
        gamma (_type_):  계절성에 대한 가중치
        seasonality_period (_type_): 데이터 주기적 패턴이 발생하는 주기

    Returns:
        _type_: 삼중 지수 평활법 적용한 최종 값
    """
    result = []
    seasonals, season_averages = [], []
    m = seasonality_period

    # 초기 추정값
    for i in range(m):
        seasonals.append(np.mean(data[i::m]))
    for i in range(len(data) + 1):
        if i == 0:
            level, trend = data[0], data[1] - data[0]
            result.append(data[0])
            season_averages.append(seasonals[i % m])
        elif i == len(data):
            result.append(level + trend * i + seasonals[i % m])
        else:
            obsval = data[i]
            last_seasonal, season_avg = seasonals[i % m], season_averages[i - m]
            level, trend = triple_exponential_update(obsval, level, trend, last_seasonal, season_avg, alpha, beta, gamma, m)
            seasonals[i % m] = gamma * (obsval - (level + trend)) + (1 - gamma) * last_seasonal
            season_averages.append(season_averages[i - m] + gamma * (obsval - (level + trend)) - last_seasonal)

            result.append(level + trend + seasonals[i % m])

    return result

def triple_exponential_update(obsval, level, trend, last_seasonal, season_avg, alpha, beta, gamma, m):

    level = alpha * (obsval - last_seasonal) + (1 - alpha) * (level + trend)
    trend = beta * (level - (level + season_avg)) + (1 - beta) * trend
    return level, trend


def draw_plot(detect_count, smoothed_data, smooth_value, saved_name):
    """
    smoothing 기법 적용 결과 plot 생성 및 저장

    Args:
        detect_count (_type_): 프레임 별 YOLOv7 탐지된 갯수
        smoothed_data (_type_): 해당 smoothing 기법을 적용한 결과 데이터
        smooth_value (_type_): 해당 smoothing 기법 적용 값
        saved_name (_type_): smoothing 기법 적용 결과 파일 저장 경로 및 이름
    """
    # 그래프 그리기
    plt.plot(detect_count, label='origin')
    plt.plot(smoothed_data, label=f'{smooth_value} moving average')

    # 그래프에 레이블과 범례 추가
    plt.xlabel('data point')
    plt.ylabel('value')
    plt.title('moving average')
    plt.legend()
    
    plt.savefig(saved_name)
    plt.clf()

def draw_smoothing(detect_count, dir):
    """
    4가지 smoothing 기법 적용

    Args:
        detect_count (_type_): 프레임 별 YOLOv7 탐지된 갯수
        dir (_type_): smoothing 기법 적용 결과 저장 경로
    """

    # moving_average_smoothing
    window_size = 180
    smoothed_data = moving_average(detect_count, window_size)
    
    draw_plot(detect_count, smoothed_data, window_size, dir + 'moving_average')
    
    # exponential_smoothing
    alpha = 0.2
    smoothed_data = exponential_smoothing(detect_count, alpha)
    draw_plot(detect_count, smoothed_data, alpha, dir + 'exponential')
    
    # double exponential_smoothing
    alpha = 0.2
    beta = 0.2
    smoothed_data = double_exponential_smoothing(detect_count, alpha, beta)
    draw_plot(detect_count, smoothed_data, alpha, dir + 'double_exponential')
    

    # Triple exponential_smoothing
    alpha = 0.1
    beta = 0.2
    gamma = 0.2
    seasonality_period = 2

    smoothed_data = triple_exponential_smoothing(detect_count, alpha, beta, gamma, seasonality_period)
    draw_plot(detect_count, smoothed_data, alpha, dir + 'triple_exponential')


if __name__=="__main__":
    
    detect_count = read_file()
    
    draw_smoothing(detect_count, ROOT + FOLDER_NAME)
    
    