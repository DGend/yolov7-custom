import argparse
from re import T
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

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
from matplotlib.animation import FuncAnimation
import datetime
import pickle
import datetime
from tqdm import tqdm

import keyboard

fig, ax = plt.subplots()

# 전역 변수 또는 클래스 변수로 vid_path와 vid_writer를 관리
vid_path = None
vid_writer = None

"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat] if colors else None
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl) # type: ignore

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img

########################################################

class VideoProcessor:
    def __init__(self, opt, save_interval_minute=2):
        self.opt = opt
        self.vid_path = None
        self.vid_writer = None
        
        # 모델 및 장치 설정 관련 초기화
        self.source = opt.source
        self.weights = opt.weights
        self.view_img = opt.view_img
        self.save_txt = opt.save_txt
        self.imgsz = opt.img_size
        self.trace = not opt.no_trace
        self.device = select_device(opt.device)
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        self.save_img = not opt.nosave and not self.source.endswith('.txt')
        # self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        # ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # 추적 관련 설정
        self.model, self.stride, self.imgsz = self.initialize_model()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        # 데이터 로더 초기화
        self.dataset = self.initialize_dataloader()
        
        # 프레임 저장 설정
        self.df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
        self.frame_id = 0
        self.save_interval_minute = save_interval_minute # default: 2분 간격으로 데이터 저장

    def initialize_model(self):
        # Load and configure model
        model = attempt_load(self.weights, map_location=self.device)
        stride = int(model.stride.max())
        imgsz = check_img_size(self.imgsz, s=stride)
        
        if self.trace:
            model = TracedModel(model, self.device, imgsz)
        
        if self.device.type != 'cpu':
            model.half()  # Use half precision if available
        
        return model, stride, imgsz

    def initialize_dataloader(self):
        # Setup dataloader for images or streams        
        if self.opt.use_live_camera:
            dataset = LoadWebcam(img_size=self.imgsz, stride=self.stride)
        else:
            # dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        return dataset

    def process_frame(self, img):
        # Prepare image and perform inference
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Perform inference
        pred = self.model(img, augment=self.opt.augment)[0]
        return pred, img

    def process_detections(self, pred, img_shape, sort_tracker, im0):
        """ 
        Apply non-max suppression and rescale boxes
        """
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        
        for det in pred:  # detections per image
            if len(det):
                if self.opt.use_live_camera:  # batch_size >= 1
                    det[:, :4] = scale_coords(img_shape, det[:, :4], im0.shape).round()
                else:
                    det[:, :4] = scale_coords(img_shape, det[:, :4], im0.shape).round()
                
                # Detected object classes를 포함하는 배열 생성
                dets_to_sort = np.empty((0,6))
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                # 추적이 활성화된 경우 track_objects 호출
                if self.opt.track:
                    bbox_xyxy, identities, categories, confidences, _im0, tracks = self.track_objects(dets_to_sort, sort_tracker, im0)
                    
                    moves = self.get_track_length(tracks)
                
                    if self.df.empty:
                        self.df = pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])
                    else:
                        self.df = pd.concat([self.df, pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])])
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                    _im0 = im0

                # Boxes 그리기
                _im0 = draw_boxes(_im0, bbox_xyxy, identities, categories, confidences, self.names, self.colors)
        
        return pred, _im0

    def track_objects(self, dets_to_sort, sort_tracker, im0):
        """
        트랙커를 사용하여 객체를 추적하고, 추적된 결과를 반환하는 함수
        """
        tracked_dets = sort_tracker.update(dets_to_sort, self.opt.unique_track_color)
        tracks = sort_tracker.getTrackers()

        # 추적 결과가 있을 때만 시각화
        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            confidences = None

            if self.opt.show_track:
                for t, track in enumerate(tracks):
                    track_color = self.colors[int(track.detclass)] if not self.opt.unique_track_color else sort_tracker.color_list[t]

                    # 트랙 경로를 그립니다.
                    for i in range(len(track.centroidarr) - 1):
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

    def save_data(self, save_path):
        if self.frame_id % (self.save_interval_minute * 60) == 0:
            file_path = save_path + datetime.datetime.now().strftime('_log_%y%m%d_%H%M.pkl')
            save_log_data(self.df, file_path)
            self.df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
            self.frame_id = 0

    def check_webcam(self):
        # 실시간 웹캠 사용 여부
        if self.opt.use_live_camera:
            cap = cv2.VideoCapture(0)  # 0 represents the default webcam index

            if cap.isOpened():
                print("Webcam is connected")
                cap.release()
                return True
            else:
                print("Webcam is not connected")
            
        return False

    def show_stream(self, im0):
        # Stream results
        ######################################################
        if self.dataset.mode != 'image' and self.opt.show_fps:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime
            cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

        #######################################################
        if self.view_img:
            cv2.imshow('', im0)
            cv2.waitKey(1)  # 1 millisecond

    def save_result(self, im0, vid_cap, save_path):
        # Save results (image with detections)
        if self.dataset.mode == 'image':
            _save_path = save_path + '.jpg'
            cv2.imwrite(_save_path, im0)
            print(f"The image with the result is saved in: {_save_path}")
        else:  # 'video' or 'stream'
            if self.vid_path != save_path:  # new video
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                _save_path = save_path + '.mp4'
                self.vid_writer = cv2.VideoWriter(_save_path, cv2.VideoWriter_fourcc(*'X264'), fps, (w, h))
            self.vid_writer.write(im0)

    def run(self):
        if self.check_webcam():
            if self.opt.use_live_camera:
                self.run_webcam()
            else:
                print("You use --use_live_webcam True, but webcam is not connected!")
                return
        else:
            self.run_video()
        
        #TODO: plot_html 파일
        # plot_timeline_ids()
    
    def run_webcam(self):
        if self.opt.use_live_camera:
            self.source = 'live_camera_' + datetime.datetime.now().strftime("%y%m%d") + '.mp4'

        if not self.opt.nosave:
            (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        
        for path, img, im0s, vid_cap in self.dataset:
            pred, img = self.process_frame(img)
            pred, _im0s = self.process_detections(pred, img.shape[2:], sort_tracker, im0s)
            
            p = os.path.splitext(Path(path).name)[0]
            save_path = str(self.save_dir / p)
            
            # Process results and save data
            self.save_data(save_path)

            # Display or save the output
            self.show_stream(_im0s)
            
            if self.save_img:
                self.save_result(_im0s, vid_cap, save_path)

            self.frame_id += 1
        
            # 'q' 키를 누르면 루프에서 나갑니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자가 'q'를 눌러 웹캠 스트리밍을 종료했습니다.")
                break
            
            if keyboard.is_pressed('q'):
                print("사용자가 'q'를 눌러 웹캠 스트리밍을 종료했습니다.")
                break
        
        self.vid_writer.release()
    
    def run_video(self):
        if not self.opt.nosave:
            (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        
        for path, img, im0s, vid_cap in self.dataset:
            pred, img = self.process_frame(img)
            pred, _im0s = self.process_detections(pred, img.shape[2:], sort_tracker, im0s)
            
            p = os.path.splitext(Path(path).name)[0]
            save_path = str(self.save_dir / p)  # img.jpg
            
            # Process results and save data
            self.save_data(save_path)

            # Display or save the output
            self.show_stream(_im0s)
            
            if self.save_img:
                self.save_result(_im0s, vid_cap, save_path)

            self.frame_id += 1
        
        self.vid_writer.release()
    
    def get_track_length(self, tracks):
        infos = []
        
        for track in tracks:
            length = len(track.centroidarr)
            infos.append({'ID': track.id, 'Length': length, 'creation_time': datetime.datetime.now()})
            
        return infos

###############################################################################################################

def detect(frame=30, save_interval_minite=2):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    
    # 웹캠 사용 여부 확인
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')) or opt.use_live_camera
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    
    df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
    
    frame_id = 0
    save_interval = save_interval_minite * frame * 60 # 분 단위로 저장할 프레임 간격 설정
    
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    
    # select cpu or cuda
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, use_live_camera=opt.use_live_camera)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, conf, detclass])))


                if opt.track:
                    
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                                
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                                int(track.centroidarr[i+1][1])),
                                                track_color, thickness=opt.thickness) 
                                                for i,_ in  enumerate(track.centroidarr) 
                                                    if i < len(track.centroidarr)-1 ] 
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                
                moves = get_track_length(tracks)
                
                if df.empty:
                    df = pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])
                else:
                    df = pd.concat([df, pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])])
                
                # 설정된 인터벌마다 DataFrame을 저장하고 새로운 DataFrame 시작
                if frame_id % save_interval == 0:
                    # file_name = datetime.now().strftime('data_%y%m%d_%H%M%S.pkl')  # 파일명 형식 변경
                    file_path = os.path.splitext(save_path)[0] + datetime.datetime.now().strftime('data_%y%m%d_%H%M.pkl')
                    # with open(file_path, 'wb') as file:
                    #     pickle.dump(df, file)
                    
                    save_log_data(df, file_path)
                    
                    # 초기화
                    df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
                    frame_id = 0

                frame_id += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
        # 마지막으로 남은 데이터 저장
        file_path = os.path.splitext(save_path)[0] + datetime.datetime.now().strftime('data_%y%m%d_%H%M.pkl')
        save_log_data(df, file_path)
        
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        
    print(f'Done. ({time.time() - t0:.3f}s)')

def detect_webcam(frame=30, save_interval_minite=2):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    # 실시간 웹캠 사용 여부
    if opt.use_live_camera:
        cap = cv2.VideoCapture(0)  # 0 represents the default webcam index

        if cap.isOpened():
            print("Webcam is connected")
            source = 'live_camera_' + datetime.datetime.now().strftime("%y%m%d") + '.mp4'
        else:
            print("Webcam is not connected")
            raise "You use --use_live_webcam True, but webcam is not connected!"

        cap.release()
    
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    
    # 웹캠 사용 여부 확인
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')) or opt.use_live_camera
    
    
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    
    df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
    
    frame_id = 0
    save_interval = save_interval_minite * frame * 60 # 분 단위로 저장할 프레임 간격 설정
    
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    
    # select cpu or cuda
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, use_live_camera=opt.use_live_camera)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, conf, detclass])))


                if opt.track:
                    
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                                
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                                int(track.centroidarr[i+1][1])),
                                                track_color, thickness=opt.thickness) 
                                                for i,_ in  enumerate(track.centroidarr) 
                                                    if i < len(track.centroidarr)-1 ] 
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                
                moves = get_track_length(tracks)
                
                if df.empty:
                    df = pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])
                else:
                    df = pd.concat([df, pd.DataFrame(moves, columns=['ID', 'Length', 'creation_time'])])
                
                # 설정된 인터벌마다 DataFrame을 저장하고 새로운 DataFrame 시작
                if frame_id % save_interval == 0:
                    # file_name = datetime.now().strftime('data_%y%m%d_%H%M%S.pkl')  # 파일명 형식 변경
                    file_path = os.path.splitext(save_path)[0] + datetime.datetime.now().strftime('data_%y%m%d_%H%M.pkl')
                    # with open(file_path, 'wb') as file:
                    #     pickle.dump(df, file)
                    
                    save_log_data(df, file_path)
                    
                    # 초기화
                    df = pd.DataFrame(columns=['ID', 'Length', 'creation_time'])
                    frame_id = 0

                frame_id += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
        # 마지막으로 남은 데이터 저장
        file_path = os.path.splitext(save_path)[0] + datetime.datetime.now().strftime('data_%y%m%d_%H%M.pkl')
        save_log_data(df, file_path)
        
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        
    print(f'Done. ({time.time() - t0:.3f}s)')

def save_log_data(df, save_path):
    grouped = df.groupby('creation_time').agg({'ID': list, 'Length': list}).reset_index()
    
    # Open the file in binary mode and save the 'grouped' object as a pickle
    with open(save_path, 'wb') as file:
        pickle.dump(grouped, file)

def count_num_fish(bbox_xyxy):
    # 현 프레임에서 탐지된 물고기 수 반환
    return len(bbox_xyxy)

def get_track_length(tracks):
    infos = []
    
    for track in tracks:
        length = len(track.centroidarr)
        infos.append({'ID': track.id, 'Length': length, 'creation_time': datetime.datetime.now()})
        
    return infos


def init_video_plot():
    global fig, ax
    # 그래프 초기화
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    
    return line

def update(df, line):
    line.set_ydata(len(df))
    return line

def update_video_plot(data):
    global fig, ax
    
    # 애니메이션 객체 생성
    ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='halibut_ver2.pt', help='model.pt path(s)')
    
    # parser.add_argument('--source', type=str, default='data/data/Infrared/test', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/data/Infrared/feed_summery.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/data/Infrared/feed_summery_20sec.mp4', help='source')  # file/folder, 0 for webcam
    
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', type=bool, default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', type=int, nargs='+', help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', default=True, help='run tracking')
    parser.add_argument('--show-track', default=True, help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=5, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    parser.add_argument('--save_log_data', default='./log_data.pickle', help='save log data to pickle file')
    
    parser.add_argument('--use_live_camera', default=True, help='use live camera') # 웹캠 사용 여부
    
    # TODO: 먹이 급이 설정 추가
    parser.add_argument('--cutoffood', type=float, default=800.0, help='food') # 먹이 급이 중단 설정

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5, # 오브젝트가 사라지기 전까지의 저장할 프레임 수
                       min_hits=2, # 오브젝트를 추적하기 위한 최소 히트 수
                       iou_threshold=0.2, # 탐지된 물체가 같은 물체인지 판단하는 IOU 임계값
                       max_id=300 # 오브젝트를 동시에 추적할 수 있는 최대 ID 수
                    )

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        video = VideoProcessor(opt)
        video.run()
        
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        #     video = VideoProcessor(opt)
        #     video.run()
            
        #     # detect()
