import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from ultralytics import YOLO
from kalman import KalmanFilter
from iou import bbox_iou
from scipy.optimize import linear_sum_assignment
import torchreid
import torch
import torchvision.transforms as T
from PIL import Image

# -----------------------------
# 初始化 ReID 模型
# -----------------------------
reid_model = torchreid.models.build_model(
    name='osnet_x0_25', num_classes=1000, pretrained=True
)
reid_model.to("cuda")
reid_model.eval()

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embeddings_batch(patches, device="cuda"):
    tensors = torch.stack([transform(p) for p in patches]).to(device)
    with torch.no_grad():
        feats = reid_model(tensors)
    feats = feats.cpu().numpy()
    
    feats = feats[:, :128]  # 截取前 128 維
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
    return feats

# ----------------------------
# Track 類別
# ----------------------------
class Track:
    def __init__(self, bbox, track_id, feature=None):
        self.kf = KalmanFilter()
        self.kf.initiate(bbox)
        self.bbox = bbox
        self.track_id = track_id
        self.time_since_update = 0
        self.features_history = []
        if feature is not None:
            self.features_history.append(feature)
        self.feature = feature

    def update_feature(self, feat, max_history=10):
        self.features_history.append(feat)
        if len(self.features_history) > max_history:
            self.features_history.pop(0)
        weights = np.arange(1, len(self.features_history)+1)
        weighted_feats = np.array([f*w for f,w in zip(self.features_history, weights)])
        self.feature = np.sum(weighted_feats, axis=0) / np.sum(weights)

# ----------------------------
# BoT-SORT
# ----------------------------
class BoTSORT:
    def __init__(self, max_age=100, iou_threshold=0.5, alpha_iou=0.5, alpha_feat=0.5, det_conf_threshold=0.6):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.alpha_iou = alpha_iou
        self.alpha_feat = alpha_feat
        self.det_conf_threshold = det_conf_threshold

    def update(self, detections):
        # 預測現有 track
        for track in self.tracks:
            track.bbox = track.kf.predict()
            track.time_since_update += 1

        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(detections)))
            matches = []
        else:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, (bbox, det_feat, conf) in enumerate(detections):
                    # IOU cost
                    iou_cost = 1 - bbox_iou(track.bbox, bbox)

                    # Appearance cost
                    if track.feature is not None and det_feat is not None:
                        feat_cost = 1 - np.dot(track.feature, det_feat)
                    else:
                        feat_cost = 1.0

                    # 動態權重: 根據 detection conf 與 track 狀態來平衡
                    # conf 越高，越相信外觀；conf 越低，越相信 IOU
                    conf_weight = conf
                    time_weight = min(track.time_since_update, 5) / 5.0
                    # conf 越高 → 越靠 IOU
                    alpha_iou = self.alpha_iou + conf_weight * 0.3 + time_weight * 0.2
                    alpha_feat = self.alpha_feat + (1 - conf_weight) * 0.3 - time_weight * 0.1

                    # normalize
                    total = alpha_iou + alpha_feat
                    alpha_iou /= total
                    alpha_feat /= total


                    cost_matrix[i, j] = alpha_iou * iou_cost + alpha_feat * feat_cost

            row, col = linear_sum_assignment(cost_matrix)
            matches, unmatched_tracks, unmatched_dets = [], [], []
            for r, c in zip(row, col):
                if cost_matrix[r, c] > (1 - self.iou_threshold):
                    unmatched_tracks.append(r)
                    unmatched_dets.append(c)
                else:
                    matches.append((r, c))
            unmatched_tracks += list(set(range(len(self.tracks))) - set(row))
            unmatched_dets += list(set(range(len(detections))) - set(col))

        # 更新匹配到的 track
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            bbox, det_feat, conf = detections[d_idx]
            track.kf.update(bbox)
            track.bbox = bbox
            track.update_feature(det_feat)
            track.time_since_update = 0

        # 創建新的 track
        for d_idx in unmatched_dets:
            bbox, det_feat, conf = detections[d_idx]
            if conf >= self.det_conf_threshold:
                self.tracks.append(Track(bbox, self.next_id, feature=det_feat))
                self.next_id += 1

        # 移除過老 track
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [(t.track_id, t.bbox) for t in self.tracks]


# ----------------------------
# 顏色工具
# ----------------------------
def get_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 256, size=3).tolist())

# ----------------------------
# Main
# ----------------------------
video_path = "videos/hard.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("outputs/output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

yolo_model_path = "models/yolo11x.pt"
yolo_model = YOLO(yolo_model_path)
yolo_model.to("cuda")

tracker = BoTSORT(det_conf_threshold=0.5)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = yolo_model.predict(frame, device="cuda", stream=True)

    # ----------------------------
    # 收集 patch 批次
    # ----------------------------
    patches, bboxes, conf_list_keep = [], [], []
    for res in results:
        cls_list = res.boxes.cls.cpu().numpy()
        box_list = res.boxes.xywh.cpu().numpy()
        conf_list = res.boxes.conf.cpu().numpy()
        for cls, box, conf in zip(cls_list, box_list, conf_list):
            if int(cls) == 0 and conf >= tracker.det_conf_threshold:
                cx, cy, w, h = box
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                patch = frame[y1:y2, x1:x2]
                patches.append(Image.fromarray(patch[:, :, ::-1]))
                bboxes.append([x1, y1, x2, y2])
                conf_list_keep.append(conf)

    # ----------------------------
    # ReID 特徵
    # ----------------------------
    if len(patches) > 0:
        feats = get_embeddings_batch(patches)
        detections = list(zip(bboxes, feats, conf_list_keep))
    else:
        detections = []

    print(f"Frame {frame_count}: {len(detections)} detections")

    # ----------------------------
    # 更新 tracker
    # ----------------------------
    tracks = tracker.update(detections)

    # 畫框與 ID
    # 畫框與 ID，包括預測位置
    for track in tracker.tracks:
        color = get_color(track.track_id)
        x1, y1, x2, y2 = map(int, track.bbox)

        if track.time_since_update == 0:
            # 有匹配 detection，畫實線框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # 預測位置，畫虛線框表示 Kalman 預測
            thickness = 2
            gap = 5
            # 上下邊
            for i in range(x1, x2, gap*2):
                cv2.line(frame, (i, y1), (min(i+gap, x2), y1), color, thickness)
                cv2.line(frame, (i, y2), (min(i+gap, x2), y2), color, thickness)
            # 左右邊
            for j in range(y1, y2, gap*2):
                cv2.line(frame, (x1, j), (x1, min(j+gap, y2)), color, thickness)
                cv2.line(frame, (x2, j), (x2, min(j+gap, y2)), color, thickness)
            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    out.write(frame)
    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking finished. Output saved as outputs/output.mp4")

