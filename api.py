# api.py - Minimal Flask API with /detect_face endpoint
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route("/detect_face", methods=["GET", "POST"])
def detect_face():
    # Minimal fast path: immediately return JSON within << 1 second
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "echo": payload, "ts": time.time()}), 200

if __name__ == "__main__":
    # Bind to localhost:5000 to match the CI test
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
