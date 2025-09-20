# api.py
import os, time
from flask import Flask, request, jsonify

SAFE_MODE = os.getenv("SAFE_API", "0") == "1"

if not SAFE_MODE:
    try:
        import face_recognition  # 或 cv2, dlib 視你的實作而定
    except Exception:
        SAFE_MODE = True  # 套件缺失時自動退回安全模式

app = Flask(__name__)

@app.route("/detect_face", methods=["GET","POST"])
def detect_face():
    if SAFE_MODE:
        return jsonify({"ok": True, "mode": "safe", "ts": time.time()}), 200

    # ← 真實人臉邏輯（範例：multipart 上傳 'file'）
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "no file"}), 400
    import numpy as np
    from PIL import Image
    img = Image.open(file.stream).convert("RGB")
    arr = np.array(img)
    boxes = face_recognition.face_locations(arr, model="hog")
    return jsonify({"ok": True, "mode": "real", "faces": len(boxes)}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
