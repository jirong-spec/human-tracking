import numpy as np

class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        # 狀態向量: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros((8,1))
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = dt  # x = x + vx*dt
        self.H = np.eye(4,8)  # 只能量測 [cx, cy, w, h]
        self.P = np.eye(8) * 10.0
        self.Q = np.diag([0.01]*4 + [0.1]*4)
        self.R = np.eye(4) * 1.0

    def initiate(self, bbox):
        x, y, w, h = bbox
        self.x[:4] = np.array([[x],[y],[w],[h]])
        self.x[4:] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()

    def update(self, bbox, confidence=1.0):
        z = np.array(bbox).reshape((4,1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y * confidence
        self.P = (np.eye(8) - K @ self.H) @ self.P

