import numpy as np
import cv2

def extract_skeleton_from_video(video_path):
    """
    여기에서는 이미 skeleton video가 ST-GCN 형식으로 만들어져 있다고 가정.
    즉, 좌표 numpy가 이미 준비되어 있음.
    실제 NTU skeleton extraction이 필요하면 알려줘.
    """

    # 여기서는 dummy load 대신 너의 파일 규칙 기반으로 load
    # sample → (3,T,25,2) 형태
    npy_path = video_path.replace(".mp4", ".npy")

    data = np.load(npy_path)  # ST-GCN 표준 구조
    return data
