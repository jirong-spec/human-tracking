import numpy as np
from kalman import KalmanFilter
from iou import bbox_iou
from scipy.optimize import linear_sum_assignment

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
                    iou_cost = 1 - bbox_iou(track.bbox, bbox)
                    feat_cost = 1 - np.dot(track.feature, det_feat) if (track.feature is not None and det_feat is not None) else 1.0

                    # 取得權重
                    alpha_iou = self.alpha_iou 
                    alpha_feat = self.alpha_feat 
                    
                    # 歸一化
                    total = alpha_iou + alpha_feat + 1e-6
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

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            bbox, det_feat, conf = detections[d_idx]
            track.kf.update(bbox)
            track.bbox = bbox
            track.update_feature(det_feat)
            track.time_since_update = 0

        for d_idx in unmatched_dets:
            bbox, det_feat, conf = detections[d_idx]
            if conf >= self.det_conf_threshold:
                self.tracks.append(Track(bbox, self.next_id, feature=det_feat))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [(t.track_id, t.bbox) for t in self.tracks]




