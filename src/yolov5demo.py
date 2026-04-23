import argparse
from array import array
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import torch
import torchvision.ops.boxes as bops
import yolov5

import norfair
from norfair import Detection, Tracker, Video, Paths

import time
import os
import random
import cv2

import warnings
warnings.filterwarnings("ignore", message="torch.cuda.amp.autocast")

seed = 123
random.seed(seed)
np.random.seed(seed)

DISTANCE_THRESHOLD_BBOX: float = 3.33
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

tracks_history = {}
tracks_history_filtered = {}

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(0,255,3).tolist())

class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def center(points):
    return [np.mean(np.array(points), axis=0)]


def iou_pytorch(detection, tracked_object):
    # Slower but simplier version of iou

    detection_points = np.concatenate([detection.points[0], detection.points[1]])
    tracked_object_points = np.concatenate(
        [tracked_object.estimate[0], tracked_object.estimate[1]]
    )

    box_a = torch.tensor([detection_points], dtype=torch.float)
    box_b = torch.tensor([tracked_object_points], dtype=torch.float)
    iou = bops.box_iou(box_a, box_b)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return np.float(1 / iou if iou else MAX_DISTANCE)


def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return 1 / iou if iou else (MAX_DISTANCE)


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
    track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores)
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores)
            )

    return norfair_detections

def paint_tracks(frame_tracks, output_path:str, tracks:dict, name:str = "tracks"):
        for track_id, points in tracks.items():
            color = get_color(track_id)
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))

                cv2.line(frame_tracks, pt1, pt2, color, 2)
            #cv2.circle(frame_tracks, points[-1], 4, color, -1)

        #print(*tracks_history.values(), sep = "\n")
        #print(len(tracks_history))
        tracks_path = os.path.join(os.path.dirname(output_path),name, os.path.basename(output_path).replace(".mp4", "_"+name+".png"))
        print(tracks_path)
        os.makedirs(os.path.dirname(tracks_path), exist_ok=True)

        cv2.imwrite(tracks_path, frame_tracks)

        print(f"Saved: '{tracks_path}'")

def add_noise(csv_path, tracks_points, all_tracks_points, p: float = 10):
    tracks_points = np.asarray(tracks_points)
    all_tracks_points = np.asarray(all_tracks_points)

    n = round((len(tracks_points) * (p / 100)))

    tracks_set = {tuple(point) for point in tracks_points}

    remaining_points = []
    for point in all_tracks_points:
        t = tuple(point)
        if t not in tracks_set:
            remaining_points.append(point)  # keep as array

    n = min(n, len(remaining_points))

    noise_points = np.vstack(random.sample(remaining_points, n))

    arrayCSV_noise = np.vstack((tracks_points, noise_points))
    folder = os.path.join(os.path.dirname(csv_path),f"{p}_noise")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder,os.path.basename(csv_path))
    pd.DataFrame(arrayCSV_noise).to_csv(filename, index=False)

    print(f"Saved: '{filename}'")


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="input/yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument("--classes", nargs="+", type=int, help="Filter by class: --classes 0, or --classes 0 2 3")
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
parser.add_argument("--age", type=int, default="100", help="Edad a partir de la que se considera un vehículo como apto para guardar su trayectoria")
parser.add_argument("--numVehiculos", type=int, default="200", help="Número de vehículos máximo a detectar (es recomendable poner más del valor máximo)")
parser.add_argument("--also_initials", type=bool, default=True, help="También registrar posiciones iniciales")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

# Creacion del array
rows, cols = (args.numVehiculos, 2)
arrayDatos = [[0 for i in range(cols)] for j in range(rows)]
candidates = dict()
initials = dict()
ultID = 0

#for input_path in args.files:
for input_path, output_path, csv_path in zip(*(iter(args.files),) * 3):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = Video(input_path=input_path, output_path=output_path)

    last_frame = None
    
    runtime_detection_s = 0.0
    runtime_tracking_s = 0.0
    num_frames = 0

    distance_function = iou if args.track_points == "bbox" else euclidean_distance
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if args.track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )
    paths_drawer = Paths(center, attenuation=0.01)

    for frame in video:
        last_frame = frame.copy()
        num_frames += 1

        t0 = time.perf_counter()
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thresh,
            image_size=args.img_size,
            classes=args.classes
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_detection_s += time.perf_counter() - t0
        
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
        
        t1 = time.perf_counter()
        tracked_objects = tracker.update(detections=detections)
        runtime_tracking_s += time.perf_counter() - t1

        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
            norfair.draw_tracked_boxes(frame, tracked_objects)

        frame = paths_drawer.draw(frame, tracked_objects)

        for obj in tracked_objects:

            if args.track_points == "centroid":

                val0 = obj.last_detection.points[0]
                val1 = obj.last_detection.points[1]

            elif args.track_points == "bbox":    
                
                # Calcular el centro del vehículo
                centerCoord = ((obj.last_detection.points[0][0]+obj.last_detection.points[1][0])/2, (obj.last_detection.points[0][1]+obj.last_detection.points[1][1])/2)

                val0 = centerCoord[0]
                val1 = centerCoord[1]
            if args.also_initials and obj.id not in candidates:
              candidates[obj.id] = (val0, val1)
            if(obj.age >= args.age):
                arrayDatos[obj.id-1] = (val0, val1)

                if args.also_initials and obj.id not in initials:
                  initials[obj.id] = candidates[obj.id]

                if(ultID < obj.id):
                    ultID = obj.id
                
                # guardar los tracks de los puntos iniciales y finales guardados en el csv (los que se usan para el clustering)
                if obj.id not in tracks_history_filtered:
                    tracks_history_filtered[obj.id] = []

                tracks_history_filtered[obj.id].append((val0, val1))

            # guardar todos los tracks (independientemente de si cumplen o no el age establecido). Sirve para introducir puntos de ruido para simular errores del tracker
            if obj.id not in tracks_history:
                tracks_history[obj.id] = []

            tracks_history[obj.id].append((val0, val1))

            del val0, val1

        video.write(frame)
    frame_tracks = last_frame.copy()
    paint_tracks(frame_tracks, output_path, tracks_history_filtered, name = "tracks")
    paint_tracks(frame_tracks, output_path, tracks_history, name = "all_tracks")

    arrayCSV = np.asarray(arrayDatos[0:ultID][:])
    arrayCSV = arrayCSV[~np.all(arrayCSV == 0, axis=1)]
    arrayCSV = np.vstack((arrayCSV, np.asarray([initials[k] for k in initials])))
    # np.savetxt('resultadosYolov5.csv',arrayCSV, fmt = '%d', delimiter=",")   
    #csv_path = 'resultadosYolov5.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(arrayCSV).to_csv(csv_path, index=False)
    print(f"Saved: '{csv_path}'")

    noise = [5,10,20]
    for p in noise:
        arrayCSV_noise = add_noise(csv_path = csv_path, tracks_points = arrayCSV, all_tracks_points = np.vstack(list(tracks_history.values())), p = p)
    

    video_name = os.path.basename(input_path)
    df_detection_new = pd.DataFrame([{
        "video": video_name,
        "network": os.path.basename(args.detector_path),
        "frames": num_frames,
        "runtime_total_s": runtime_detection_s,
        "runtime_per_frame_s": runtime_detection_s / num_frames,
        "runtime_total_ms": runtime_detection_s * 1000.0,
        "runtime_per_frame_ms": (runtime_detection_s / num_frames) * 1000.0
    }])

    df_tracking_new = pd.DataFrame([{
        "video": video_name,
        "network": os.path.basename(args.detector_path),
        "frames": num_frames,
        "runtime_total_s": runtime_tracking_s,
        "runtime_per_frame_s": runtime_tracking_s / num_frames,
        "runtime_total_ms": runtime_tracking_s * 1000.0,
        "runtime_per_frame_ms": (runtime_tracking_s / num_frames) * 1000.0
    }])

    output_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)),"runtime")
    os.makedirs(output_dir, exist_ok=True)

    csv_detection = os.path.join(output_dir,"runtime_detection.csv")
    if os.path.exists(csv_detection):
        df_detection_old = pd.read_csv(csv_detection)
        df_detection = pd.concat([df_detection_old, df_detection_new], ignore_index=True)
    else:
        df_detection = df_detection_new

    df_detection.to_csv(csv_detection, index=False)
    print(f"Saved: '{csv_detection}'")

    csv_tracking = os.path.join(output_dir,"runtime_tracking.csv")
    if os.path.exists(csv_tracking):
        df_tracking_old = pd.read_csv(csv_tracking)
        df_tracking = pd.concat([df_tracking_old, df_tracking_new], ignore_index=True)
    else:
        df_tracking = df_tracking_new

    df_tracking.to_csv(csv_tracking, index=False)
    print(f"Saved: '{csv_tracking}'")

    txt_detection = csv_detection.replace("csv","txt")
    txt_tracking = csv_tracking.replace("csv","txt")

    with open(txt_detection, "w") as f:
        f.write(df_detection.to_latex(index=False, float_format="%.4f", escape = True))
    print(f"Saved: '{txt_detection}'")

    with open(txt_tracking, "w") as f:
        f.write(df_tracking.to_latex(index=False, float_format="%.4f", escape = True))
    print(f"Saved: '{txt_tracking}'")



