# tests/run_api_test.py
import argparse, os, time, subprocess, signal, socket, sys, json
from contextlib import closing
import requests

def wait_port(host, port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(0.2)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="api.py", help="Your Flask entry file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--endpoint", default="/detect_face")
    ap.add_argument("--log", default="api_test.log")
    ap.add_argument("--timeout", type=float, default=1.0, help="Max allowed seconds for one request")
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "production")

    proc = subprocess.Popen([sys.executable, args.entry],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)

    method_used = "GET"
    status = -1
    text = ""
    elapsed = -1.0
    try:
        assert wait_port(args.host, args.port, timeout=25), "Server didn't start"
        url = f"http://{args.host}:{args.port}{args.endpoint}"

        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=2.0)
            method_used = "GET"
            status = r.status_code
            text = (r.text or "")[:400]
            if status == 405:
                t0 = time.perf_counter()
                r = requests.post(url, json={"ping": "pong"}, timeout=2.0)
                method_used = "POST"
                status = r.status_code
                text = (r.text or "")[:400]
        except requests.RequestException as e:
            status = -1
            text = f"request error: {e}"
        elapsed = time.perf_counter() - t0

        passed = (status in (200, 201)) and (elapsed < args.timeout)

        with open(args.log, "w", encoding="utf-8") as f:
            f.write(f"url={url}\n")
            f.write(f"method={method_used}\n")
            f.write(f"status={status}\n")
            f.write(f"elapsed_sec={elapsed:.3f}\n")
            f.write(f"body_preview={text}\n")
            f.write(f"criteria=http_2xx && elapsed<{args.timeout}s\n")
            f.write(f"result={'PASS' if passed else 'FAIL'}\n")

        assert passed, f"API test failed: status={status}, elapsed={elapsed:.3f}s"

    finally:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass

if __name__ == "__main__":
    main()
