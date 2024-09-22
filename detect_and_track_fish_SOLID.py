from abc import ABC, abstractmethod
import datetime
import argparse
import torch
from draw_smoothing import moving_average
from models.experimental import attempt_load
from utils.datasets import LoadWebcam, LoadImages, LoadStreams
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sort import *
import pandas as pd
import numpy as np
import cv2
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import threading


# 인터페이스 정의
class IModelLoader(ABC):
    """모델 로더 인터페이스."""
    @abstractmethod
    def load_model(self):
        """모델을 로드합니다.

        @return 모델 객체, stride, 이미지 크기의 튜플.
        """
        pass

class IDataLoader(ABC):
    """데이터 로더 인터페이스."""
    @abstractmethod
    def load_data(self):
        """데이터를 로드합니다.

        @return 데이터셋 객체.
        """
        pass

class IFrameProcessor(ABC):
    """프레임 프로세서 인터페이스."""
    @abstractmethod
    def process_frame(self, frame):
        """단일 프레임을 처리합니다.

        @param frame 입력 프레임.
        @return 예측 결과와 처리된 프레임의 튜플.
        """
        pass

class ITracker(ABC):
    """객체 추적기 인터페이스."""
    @abstractmethod
    def track_objects(self, detections):
        """객체를 추적합니다.

        @param detections 감지된 객체들.
        @return 추적 결과.
        """
        pass

class IDataAnalyzer(ABC):
    """데이터 분석기 인터페이스."""
    @abstractmethod
    def stop_feed_supply(self, df:pd.DataFrame, window_size:int=10, alpha:float=0.2, beta:float=0.2, gamma:float=0.2, period:int=2):
        """데이터를 분석합니다.

        @param data 분석할 데이터.
        """
        pass

class IDataSaver(ABC):
    """데이터 저장기 인터페이스."""
    @abstractmethod
    def save(self, data):
        """데이터를 저장합니다.

        @param data 저장할 데이터.
        """
        pass

class IVisualizer(ABC):
    """시각화 인터페이스."""
    @abstractmethod
    def visualize(self):
        """데이터를 시각화합니다.

        @param data 시각화할 데이터.
        """
        pass

# 구현 클래스 정의
class ModelLoader(IModelLoader):
    """모델 로딩 및 설정을 담당하는 클래스."""
    def __init__(self, weights, device, img_size, no_trace):
        """ModelLoader를 초기화합니다.

        @param weights 모델 가중치의 경로.
        @param device 사용할 디바이스 ('cpu' 또는 'cuda').
        @param img_size 입력 이미지 크기.
        @param no_trace 트레이싱 비활성화 여부.
        """
        ## @var weights
        # 모델 가중치 경로
        self.weights = weights
        
        ## @var device
        # 사용할 디바이스
        self.device = device

        ## @var img_size
        # 이미지 크기
        self.img_size = img_size

        ## @var no_trace
        # 트레이싱 사용 여부
        self.no_trace = no_trace

    def load_model(self):
        """모델을 로드하고 반환합니다.

        @return 모델 객체, stride, 이미지 크기의 튜플.
        """
        # 모델 로딩 로직 구현
        model = attempt_load(self.weights, map_location=self.device)
        stride = int(model.stride.max())
        imgsz = check_img_size(self.img_size, s=stride)
        if not self.no_trace:
            model = TracedModel(model, self.device, imgsz)
        if self.device.type != 'cpu':
            model.half()
        return model, stride, imgsz

class VideoDataLoader(IDataLoader):
    """비디오 또는 웹캠 데이터를 로드하는 클래스."""
    def __init__(self, source, img_size, stride, use_live_camera):
        """VideoDataLoader를 초기화합니다.

        @param source 데이터 소스 경로 또는 웹캠 인덱스.
        @param img_size 입력 이미지 크기.
        @param stride 모델 stride 값.
        @param use_live_camera 라이브 카메라 사용 여부.
        """
        ## @var source
        # 데이터 소스 경로 또는 웹캠 인덱스
        self.source = source

        ## @var img_size
        # 입력 이미지 크기
        self.img_size = img_size

        ## @var stride
        # 모델 stride 값
        self.stride = stride

        ## @var use_live_camera
        # 라이브 카메라 사용 여부
        self.use_live_camera = use_live_camera

    def load_data(self):
        """데이터를 로드합니다.

        @return 데이터셋 객체.
        """
        if self.use_live_camera:
            dataset = LoadWebcam(img_size=self.img_size, stride=self.stride)
        else:
            dataset = LoadImages(self.source, img_size=self.img_size, stride=self.stride)
        return dataset

class FrameProcessor(IFrameProcessor):
    """프레임을 처리하고 추론을 수행하는 클래스."""

    def __init__(self, model, device):
        """FrameProcessor를 초기화합니다.

        @param model 로드된 모델 객체.
        @param device 사용할 디바이스.
        """

        ## @var model
        # 로드된 모델 객체
        self.model = model

        ## @var device
        # 사용할 디바이스
        self.device = device

    def process_frame(self, img):
        """단일 프레임을 처리하고 추론을 수행합니다.

        @param img 입력 이미지.
        @return 예측 결과와 처리된 이미지의 튜플.
        """
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.half else img.float()    # 데이터 타입 변경
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        return pred, img

class ObjectTracker(ITracker):
    """객체 추적을 담당하는 클래스."""
    def __init__(self, opt, colors=None):
        """ObjectTracker를 초기화합니다.

        @param opt 옵션 파라미터 객체.
        """
        ## @var opt
        # 옵션 파라미터 객체
        self.opt = opt
        self.colors = colors

        ## @var sort_tracker
        # SORT 추적기 객체
        self.sort_tracker = Sort(
            max_age=5,          # 객체를 추적할 최대 프레임 수
            min_hits=2,         # 객체를 확정하기 위한 최소 감지 횟수
            iou_threshold=0.2,  # IOU 임계값
            max_id=300          # 최대 객체 ID 수
        )

    def track_objects(self, dets_to_sort, im0):
        """객체를 추적합니다.

        @param dets_to_sort 추적할 객체들의 정보.
        @param im0 원본 이미지.
        @return 바운딩 박스, ID, 카테고리, 신뢰도, 이미지, 트랙의 튜플.
        """
        tracked_dets = self.sort_tracker.update(dets_to_sort, self.opt.unique_track_color)
        tracks = self.sort_tracker.getTrackers()
        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4] # 바운딩 박스 좌표
            identities = tracked_dets[:, 8] # 객체 ID
            categories = tracked_dets[:, 4] # 객체 카테고리
            confidences = None
            if self.opt.track:
                for t, track in enumerate(tracks):
                    track_color = self.colors[int(track.detclass)] if not self.opt.unique_track_color else self.sort_tracker.color_list[t]
                    for i in range(len(track.centroidarr) - 1):
                        # 추적 경로를 그림
                        cv2.line(im0, 
                                (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                                (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1])),
                                track_color, thickness=self.opt.thickness)
        else:
            bbox_xyxy = dets_to_sort[:,:4]
            identities = None
            categories = dets_to_sort[:, 5]
            confidences = dets_to_sort[:, 4]
        return bbox_xyxy, identities, categories, confidences, im0, tracks

class DataAnalyzer(IDataAnalyzer):
    """데이터 분석을 담당하는 클래스."""
    def moving_average(self, data:pd.DataFrame, window_size:int=10):
        """데이터의 이동 평균을 계산합니다.

        @param data 입력 데이터 시리즈.
        @param window_size 이동 평균 창 크기.
        @return 이동 평균 시리즈.
        """
        if len(data) < window_size:
            return data
        return data.rolling(window=window_size).mean()

    def exponential_smoothing(self, data:pd.DataFrame, alpha:float=0.2):
        """지수 평활화를 적용합니다.

        @param data 입력 데이터 시리즈.
        @param alpha 평활화 계수.
        @return 평활화된 데이터 시리즈.
        """
        if len(data) < 2:
            return data
        return data.ewm(alpha=alpha).mean()

    def triple_exponential_smoothing(self, data: pd.DataFrame, alpha:float=0.1, beta:float=0.2, gamma:float=0.2, period:int=2):
        """삼중 지수 평활화를 적용합니다.

        @param data 입력 데이터 시리즈.
        @param alpha 레벨 평활화 계수.
        @param beta 추세 평활화 계수.
        @param gamma 계절 평활화 계수.
        @param period 계절 주기.
        """
        if len(data) < 2:
            return data
        
        result = pd.Series(index=data.index)
        result.iloc[0] = data.iloc[0]  # Initial prediction is the first data point

        # Initialize level, trend, and seasonal components
        level = data.iloc[0]
        trend = data.iloc[1] - data.iloc[0]
        seasonal = [data[i] - data[i - period] for i in range(period)]

        for t in range(1, len(data)):
            if t >= period:
                # Update level
                prev_level = level
                level = alpha * (data.iloc[t] - seasonal[t % period]) + (1 - alpha) * (prev_level + trend)

                # Update trend
                prev_trend = trend
                trend = beta * (level - prev_level) + (1 - beta) * prev_trend

                # Update seasonal component
                seasonal[t % period] = gamma * (data.iloc[t] - level) + (1 - gamma) * seasonal[t % period]

            # Calculate forecast
            result.iloc[t] = level + trend + seasonal[t % period]

        return result
    
    def stop_feed_supply(self, df:pd.DataFrame, window_size:int=10, alpha:float=0.2, beta:float=0.2, gamma:float=0.2, period:int=2, consecutive_max_count=20):
        """데이터 분석을 통해 먹이 공급 중단 여부를 결정합니다.
        Args:
        @param df 데이터프레임
        @param window_size 이동 평균 창 크기
        @param alpha 평활화 계수
        @param beta 추세 평활화 계수
        @param gamma  평활화 계수
        @param period triple_exponential_smoothing 주기

        Returns:
        @return 먹이 공급 중단 여부 (True 또는 False).
        """
        if df.empty:
            return False, None, None, None, None, None

        total_length_per_time = df.groupby('creation_time')['Length'].sum()

        # 스무딩 계산
        moving_avg = self.moving_average(total_length_per_time, window_size=window_size)
        exp_smooth = self.exponential_smoothing(total_length_per_time, alpha=alpha)
        triple_exp_smooth = self.triple_exponential_smoothing(total_length_per_time, alpha=alpha, beta=beta, gamma=gamma, period=period)
        
        # 현재 값 및 스무딩 값 가져오기
        current_value = total_length_per_time.iloc[-1]
        ma_value = moving_avg.iloc[-1]
        es_value = exp_smooth.iloc[-1]
        tes_value = triple_exp_smooth.iloc[-1]
        current_time = total_length_per_time.index[-1]

        # 현재 값이 세 가지 스무딩 값보다 모두 작은지 확인
        if current_value < ma_value and current_value < es_value and current_value < tes_value:
            # 연속 낮은 값 카운터 증가
            self.consecutive_low_count += 1
        else:
            # 카운터 초기화
            self.consecutive_low_count = 0

        # 카운터가 consecutive_max_count 이상이면 True 반환
        stop_feed = self.consecutive_low_count >= consecutive_max_count

        return stop_feed, current_time, current_value, ma_value, es_value, tes_value

class DataSaver(IDataSaver):
    """데이터 저장을 담당하는 클래스."""
    def save(self, data, path):
        """데이터를 파일에 저장합니다.

        @param data 저장할 데이터.
        @param path 저장 경로.
        """
        with open(path, 'wb') as file:
            pickle.dump(data, file)

class GraphVisualizer(IVisualizer):
    """데이터 시각화를 담당하는 클래스."""
    def __init__(self):
        """Visualizer를 초기화합니다."""
        # 데이터 저장을 위한 deque 객체 생성
        self.time_points = deque(maxlen=100)
        self.current_values = deque(maxlen=100)
        self.moving_avg_values = deque(maxlen=100)
        self.exp_smooth_values = deque(maxlen=100)
        self.triple_exp_smooth_values = deque(maxlen=100)

        # 플롯 초기화
        plt.ion()  # interactive 모드 켜기
        self.fig, self.ax = plt.subplots()
        self.line_current, = self.ax.plot([], [], label='Current Value')
        self.line_ma, = self.ax.plot([], [], label='Moving Average')
        self.line_es, = self.ax.plot([], [], label='Exponential Smoothing')
        self.line_tes, = self.ax.plot([], [], label='Triple Exp Smoothing')
        self.ax.legend()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')

    def update(self, time_point, current_value, ma_value, es_value, tes_value):
        """데이터를 업데이트합니다.

        @param time_point 현재 시각.
        @param current_value 현재 값.
        @param ma_value 이동 평균 값.
        @param es_value 지수 평활화 값.
        @param tes_value 삼중 지수 평활화 값.
        """
        # 데이터 추가
        self.time_points.append(time_point)
        self.current_values.append(current_value)
        self.moving_avg_values.append(ma_value)
        self.exp_smooth_values.append(es_value)
        self.triple_exp_smooth_values.append(tes_value)

    def visualize(self):
        """데이터를 시각화합니다.
        """

        # 데이터 업데이트
        self.line_current.set_data(self.time_points, self.current_values)
        self.line_ma.set_data(self.time_points, self.moving_avg_values)
        self.line_es.set_data(self.time_points, self.exp_smooth_values)
        self.line_tes.set_data(self.time_points, self.triple_exp_smooth_values)

        # 축 설정
        self.ax.relim()
        self.ax.autoscale_view()

        # 그래프 그리기
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Matplotlib 이벤트 루프에 시간을 양보
        plt.pause(0.1)

class VideoVisualizer(IVisualizer):
    # Bounding Box를 그리는 유틸리티 함수
    @staticmethod
    def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None, opt=None):
        """이미지에 바운딩 박스와 레이블을 그립니다.

        @param img 그릴 이미지.
        @param bbox 바운딩 박스 좌표.
        @param identities 객체 ID 리스트.
        @param categories 객체 카테고리 리스트.
        @param confidences 신뢰도 리스트.
        @param names 클래스 이름 리스트.
        @param colors 색상 리스트.
        @param opt 옵션 파라미터 객체.
        @return 바운딩 박스와 레이블이 그려진 이미지.
        """
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0

            color = colors[cat % len(colors)] if colors else [255, 0, 0]

            if not opt.nobbox:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

            if not opt.nolabel:
                label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return img

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt
        self.im0 = None

    def update(self, im0):
        self.im0 = im0
    
    def visualize(self):
        # Stream results
        ######################################################
        if self.dataset.mode != 'image' and self.opt.show_fps:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime
            # cv2.putText(self.im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

        #######################################################
        if self.opt.view_img:
            # 윈도우 이름을 지정하고 WINDOW_NORMAL 속성 설정
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', self.im0)


# VideoProcessor 클래스 재설계
class VideoProcessor:
    """비디오 프레임을 처리하고 객체 탐지 및 추적을 수행하는 클래스."""
    def __init__(self, opt):
        """VideoProcessor를 초기화합니다.

        @param opt 옵션 파라미터 객체.
        """
        self.opt = opt
        self.device = select_device(opt.device)
        self.model_loader = ModelLoader(opt.weights, self.device, opt.img_size, opt.no_trace)
        self.model, self.stride, self.imgsz = self.model_loader.load_model()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.data_loader = VideoDataLoader(opt.source, self.imgsz, self.stride, opt.use_live_camera)
        self.dataset = self.data_loader.load_data()
        self.frame_processor = FrameProcessor(self.model, self.device)
        self.tracker = ObjectTracker(opt, self.colors)
        self.analyzer = DataAnalyzer()
        self.saver = DataSaver()
        
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        self.df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])   # 추적 정보 저장 데이터프레임
        self.frame_id = 0   # 프레임 카운터
        self.save_interval_minute = 2   # 데이터 저장 간격 (분)
        self.vid_path = None    # 비디오 저장 경로
        self.vid_writer = None  # 비디오 저장 객체
        self.stop_feed_count = 0    # 먹이 공급 중단 카운터
        self.max_feed_count = {}    # 최대 먹이 공급 횟수 저장
        self.consecutive_low_count = 0

    def process_detections(self, pred, img_shape, im0):
        """탐지 결과를 처리하고 추적을 적용합니다.

        @param pred 모델의 예측 결과.
        @param img_shape 처리된 이미지의 크기.
        @param im0 원본 이미지.
        @return 바운딩 박스와 추적 정보가 그려진 이미지.
        """
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        for det in pred:  # 이미지당 탐지 결과
            if len(det):
                det[:, :4] = scale_coords(img_shape, det[:, :4], im0.shape).round()

                dets_to_sort = np.empty((0,6))
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                if self.opt.track:
                    bbox_xyxy, identities, categories, confidences, im0, tracks = self.tracker.track_objects(dets_to_sort, im0)
                    moves = self.get_track_length(tracks)
                    self.df = pd.concat([self.df, pd.DataFrame(moves)], ignore_index=True)
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]

                im0 = VideoVisualizer.draw_boxes(im0, bbox_xyxy, identities, categories, confidences, self.names, self.colors, self.opt)
        return im0

    def get_track_length(self, tracks):
        """각 트랙의 길이를 계산합니다.

        @param tracks 트랙 객체들의 리스트.
        @return 트랙 정보가 포함된 딕셔너리 리스트.
        """
        infos = []
        for track in tracks:
            length = len(track.centroidarr)
            infos.append({'ID': track.id, 'Length': length, 'creation_time': datetime.datetime.now()})
        return infos

    def save_data(self, save_path):
        """주기적으로 추적 데이터를 저장합니다.

        @param save_path 데이터 저장 경로.
        """
        if self.frame_id % (self.save_interval_minute * 60) == 0:
            file_path = save_path + datetime.datetime.now().strftime('_log_%y%m%d_%H%M.pkl')
            if len(self.df) > 1000:
                self.saver.save(self.df[1000:], file_path)
            else:
                self.saver.save(self.df, file_path)
            self.df = self.df.tail(1000)
            self.frame_id = 0

    def save_result(self, im0, vid_cap, save_path):
        """결과 이미지 또는 비디오를 저장합니다.

        @param im0 저장할 이미지.
        @param vid_cap 비디오 캡처 객체.
        @param save_path 저장 경로.
        """
        if self.dataset.mode == 'image':
            _save_path = save_path + '.jpg'
            cv2.imwrite(_save_path, im0)
        else:
            if self.vid_path != save_path:
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                _save_path = save_path + '.mp4'
                self.vid_writer = cv2.VideoWriter(_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer.write(im0)

    def run(self):
        """비디오 처리 파이프라인을 실행합니다."""
        if not self.opt.nosave:
            (self.save_dir / 'labels' if self.opt.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        
        video_visualizer = VideoVisualizer(self.dataset, self.opt)
        graph_visualizer = GraphVisualizer()

        for path, img, im0s, vid_cap in self.dataset:
            pred, img = self.frame_processor.process_frame(img)
            im0s = self.process_detections(pred, img.shape[2:], im0s)

            p = os.path.splitext(Path(path).name)[0]
            save_path = str(self.save_dir / p)

            self.save_data(save_path)

            if self.opt.view_img:
                video_visualizer.update(im0s)
                video_visualizer.visualize()
                if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
                    break

            if self.opt.save_img:
                self.save_result(im0s, vid_cap, save_path)

            self.frame_id += 1

            # 먹이 공급 중단 여부 확인 및 값 받아오기
            stop_feed_supply, current_time, current_value, ma_value, es_value, tes_value = self.analyzer.stop_feed_supply(self.df)
            
            if current_time is not None:
                graph_visualizer.update(current_time, current_value, ma_value, es_value, tes_value)
                graph_visualizer.visualize()

            if stop_feed_supply:
                print("먹이 급이가 중단되었습니다.")
                break
            
        if self.vid_writer:
            self.vid_writer.release()
        print("Processing completed.")
    
    # 종료 시 윈도우 모두 닫기
    cv2.destroyAllWindows()

# 메인 실행부
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='halibut_ver2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/data/Infrared/feed_original.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', type=int, nargs='+', help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--no-trace', action='store_true', help='don\'t trace model')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=5, help='bounding box thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed for bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don\'t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don\'t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='unique track color')
    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--use_live_camera', action='store_true', help='use live camera')
    parser.add_argument('--cutoffood', type=float, default=800.0, help='food cutoff threshold')
    parser.add_argument('--save-img', action='store_true', help='save images/videos')
    opt = parser.parse_args()

    # 만약 nosave 옵션이 설정되면 save_img를 False로 설정합니다.
    opt.save_img = not opt.nosave

    opt.track = True
    opt.view_img = True

    with torch.no_grad():
        video_processor = VideoProcessor(opt)
        video_processor.run()
