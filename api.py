# api.py
import os, time
from flask import Flask, request, jsonify

SAFE_MODE = os.getenv("SAFE_API", "0") == "1"

if not SAFE_MODE:
    # 只有本機/樹莓派實機才載入重依賴；CI 設 SAFE_API=1 就不載
    try:
        import cv2  # or face_recognition, dlib, etc.
        # ... 你的真實初始化 ...
    except Exception as e:
        # 若你想強制 CI 也能跑，不要 raise，改 SAFE fallback
        SAFE_MODE = True

app = Flask(__name__)

@app.route("/detect_face", methods=["GET", "POST"])
def detect_face():
    if SAFE_MODE:
        payload = request.get_json(silent=True) or {}
        return jsonify({"ok": True, "mode": "safe", "echo": payload, "ts": time.time()}), 200
    # 真實路徑：用到相機/人臉模型的邏輯
    # result = real_detect(...)
    return jsonify({"ok": True, "mode": "real", "ts": time.time()}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
