import os

def convert_video_to_skeleton(video_path):
    """
    MP4 영상과 같은 이름의 npy 파일을 자동으로 찾아 반환.
    예: A033_video.mp4 → A033_video.npy
    """

    # 확장자 제거: A033_video
    base = os.path.splitext(video_path)[0]

    # npy 경로 생성
    npy_path = base + ".npy"

    # 현재 폴더 내 위치 조합
    npy_full_path = os.path.join(os.path.dirname(video_path), os.path.basename(npy_path))

    if os.path.exists(npy_full_path):
        return npy_full_path
    else:
        print(f"[✗] No matching NPY found for: {video_path}")
        print(f"    Expected: {npy_full_path}")
        return None
