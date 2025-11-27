import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV Load
def load_csv(path):
    pts = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('u'):  # 헤더 스킵
                x, y = line.split(',')
                pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)

# Camera Intrinsics
fx = fy = 1086
cx, cy = 512, 384

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=float)

# Load images
img1 = cv.imread("D:/OpenCV/opencv/reconstruction_data/003.jpg")
img2 = cv.imread("D:/OpenCV/opencv/reconstruction_data/005.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("이미지 경로(data/003.jpg, data/005.jpg)를 확인하세요.")

# 이미지 크기가 다르다면 동일하게 맞추기
h = min(img1.shape[0], img2.shape[0])
w = min(img1.shape[1], img2.shape[1])
img1 = cv.resize(img1, (w, h))
img2 = cv.resize(img2, (w, h))

# Load manual 2D corresponding points
pts1 = load_csv("D:/OpenCV/opencv/reconstruction_data/003.csv")
pts2 = load_csv("D:/OpenCV/opencv/reconstruction_data/005.csv")

# Visualize 2D points on images
img1_vis = img1.copy()
img2_vis = img2.copy()

for x, y in pts1:
    cv.circle(img1_vis, (int(x), int(y)), 6, (0, 0, 255), -1)

for x, y in pts2:
    cv.circle(img2_vis, (int(x), int(y)), 6, (0, 0, 255), -1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("2D Points on 003.jpg")
plt.imshow(cv.cvtColor(img1_vis, cv.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("2D Points on 005.jpg")
plt.imshow(cv.cvtColor(img2_vis, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Compute Essential Matrix
E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.0)

print("=== Essential Matrix ===")
print(E)

# Recover Pose (Rotation R, Translation t)
_, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

print("\n=== Rotation R ===")
print(R)
print("\n=== Translation t ===")
print(t)

# Triangulation
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

pts1_norm = cv.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
pts2_norm = cv.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

pts4d = cv.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
pts3d = (pts4d[:3] / pts4d[3]).T   # (N, 3)

# Z > 0 필터링
pts3d = pts3d[pts3d[:, 2] > 0]

# 3D Visualization using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D 포인트
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], 
           c='blue', marker='o', s=20, alpha=0.6)

# Camera 0 (빨간 삼각형, origin)
ax.scatter([0], [0], [0], c='red', marker='^', s=200, 
           edgecolors='darkred', linewidth=2)
ax.text(0, -0.05, -0.1, 'cam0', color='red', fontsize=12, 
        weight='bold')

# Camera 1 (초록 삼각형)
cam1_pos = t.flatten()
ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], 
           c='green', marker='^', s=200, 
           edgecolors='darkgreen', linewidth=2)
ax.text(cam1_pos[0], cam1_pos[1]+0.05, cam1_pos[2], 'cam1', 
        color='green', fontsize=12, weight='bold')

# 축 레이블
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)

# 뷰포인트 설정
ax.view_init(elev=20, azim=-110)

# 그리드 및 배경
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n 3D Reconstruction Complete ({len(pts3d)} points)")
