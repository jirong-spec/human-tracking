import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from reid_utils import get_embeddings_batch
from bot_sort import BoTSORT

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

yolo_model = YOLO("models/yolo11x.pt").to("cuda")
tracker = BoTSORT(det_conf_threshold=0.5)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    results = yolo_model.predict(frame, device="cuda", stream=True)

    patches, bboxes, conf_list_keep = [], [], []
    for res in results:
        cls_list = res.boxes.cls.cpu().numpy()
        box_list = res.boxes.xywh.cpu().numpy()
        conf_list = res.boxes.conf.cpu().numpy()
        for cls, box, conf in zip(cls_list, box_list, conf_list):
            if int(cls) == 0 and conf >= tracker.det_conf_threshold:
                cx, cy, w, h = box
                x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                patch = frame[y1:y2, x1:x2]
                patches.append(Image.fromarray(patch[:, :, ::-1]))
                bboxes.append([x1, y1, x2, y2])
                conf_list_keep.append(conf)

    if len(patches) > 0:
        feats = get_embeddings_batch(patches)
        detections = list(zip(bboxes, feats, conf_list_keep))
    else:
        detections = []

    tracks = tracker.update(detections)

    for track in tracker.tracks:
        color = get_color(track.track_id)
        x1, y1, x2, y2 = map(int, track.bbox)
        if track.time_since_update == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # 虛線框（Kalman 預測）
            gap, thickness = 5, 2
            for i in range(x1, x2, gap*2):
                cv2.line(frame, (i, y1), (min(i+gap, x2), y1), color, thickness)
                cv2.line(frame, (i, y2), (min(i+gap, x2), y2), color, thickness)
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
