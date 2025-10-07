import cv2
import numpy as np

def get_color_hist(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # 轉 HSV，比較穩定
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 計算三通道直方圖 (H,S,V)
    hist = []
    for i in range(3):
        h = cv2.calcHist([hsv], [i], None, [32], [0, 256])  # 32 bins
        h = cv2.normalize(h, h).flatten()
        hist.extend(h)
    return np.array(hist, dtype=np.float32)

