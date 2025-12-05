from flask import Flask, render_template, request, jsonify
from model.stgcn_predict import predict_video_label
import os

app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "model/stgcn_onecyclelr.pth"
LABEL_PATH = "../data/NTU-RGB-D/x-view/small_test_label.pkl"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("🔵 /predict 호출됨")   # <-- 추가

    from utils.convert_video_to_skeleton import convert_video_to_skeleton

    if "video" not in request.files:
        print("❌ video 필드 없음")
        return jsonify({"error": "No file!"})

    file = request.files["video"]
    print("📁 받은 파일:", file.filename)

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    npy_path = convert_video_to_skeleton(save_path)
    print("🟡 변환된 NPY:", npy_path)
    
    pred, pred_idx, probs_np = predict_video_label(npy_path, model_path = MODEL_PATH)


    print("🟢 예측 결과:", pred)
    return jsonify({"pred": int(pred),
                    "pred_idx": int(pred_idx),
                    "probs": probs_np.tolist()   })

if __name__ == "__main__":
    app.run(debug=True)

