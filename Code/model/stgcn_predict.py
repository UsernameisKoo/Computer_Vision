import numpy as np
import torch
from model.stgcn_model_architecture import Model, get_edge, get_hop_distance, get_adjacency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# actions = numpy array
ACTIONS = np.array([0,1,7,8,10,11,12,27,28,29,32,40])
REVERSE_MAP = {i: ACTIONS[i] for i in range(len(ACTIONS))}

# mean/std 로드 (학습 때 저장한 값)
mean_std = np.load("model/train_mean_std.npz")
MEAN = torch.tensor(mean_std["mean"], dtype=torch.float32).to(device)
STD  = torch.tensor(mean_std["std"], dtype=torch.float32).to(device)  + 1e-5


def preprocess_npy(path):
    x = np.load(path)  # (3,T,25,1 ~ 2)
    
    # M=1 → M=2 zero pad
    if x.shape[-1] == 1:
        x = np.concatenate([x, np.zeros_like(x)], axis=-1)

    # → torch
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # normalize (학습 때와 동일)
    x = (x - MEAN) / STD

    return x  # (1,3,T,25,2)


def predict_video_label(npy_path, model_path=None):
    x = preprocess_npy(npy_path)
    # 필요한 그래프 불러오기
    edge, center = get_edge()
    hop_dis = get_hop_distance(25, edge, max_hop=1)
    A = get_adjacency(hop_dis, center, 25, max_hop=1, dilation=1)
    A = torch.tensor(A, dtype=torch.float32).to(device)

    # ACTION CLASS LIST (학습 기준)
    ACTIONS = np.array([0,1,7,8,10,11,12,27,28,29,32,40])

    num_class = len(ACTIONS)

    # 모델 생성
    model = Model(
        in_channels=3,
        num_class=num_class,
        A=A,
        edge_importance_weighting=True,
        dropout=0.2
    ).to(device)

    # checkpoint 로드
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)  # 확률
        probs_np = probs.cpu().numpy()[0]  # numpy로 변환

        pred_idx = int(out.argmax().cpu())
        true_label = REVERSE_MAP[pred_idx] + 1

    return true_label, pred_idx, probs_np
