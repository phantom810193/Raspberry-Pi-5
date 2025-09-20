# api.py
import os, time
from flask import Flask, request, jsonify

# CI 用 SAFE 模式（ci.yml 的 api-test 有設 SAFE_API=1）
SAFE_MODE = os.getenv("SAFE_API", "0") == "1"

if not SAFE_MODE:
    try:
        # 這裡放你真正需要的人臉/影像依賴
        # 例如：
        # import face_recognition
        # import cv2
        pass
    except Exception:
        # 若有失敗，自動退回 SAFE 模式，避免伺服器直接崩潰
        SAFE_MODE = True

app = Flask(__name__)

@app.route("/detect_face", methods=["GET", "POST"])
def detect_face():
    if SAFE_MODE:
        payload = request.get_json(silent=True) or {}
        return jsonify({"ok": True, "mode": "safe", "echo": payload, "ts": time.time()}), 200

    # TODO: 這裡寫你的真實人臉流程（讀圖→偵測→回傳）
    # 例：
    # file = request.files.get("file")
    # if not file:
    #     return jsonify({"ok": False, "error": "no file"}), 400
    # ... 處理 ...
    return jsonify({"ok": True, "mode": "real", "ts": time.time()}), 200

if __name__ == "__main__":
    # 用 127.0.0.1:5000，和測試腳本一致
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
