# tests/run_api_test.py
import argparse, os, time, subprocess, socket, sys
from contextlib import closing
import requests
from pathlib import Path

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
    ap.add_argument("--entry", default="api.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--endpoint", default="/detect_face")
    ap.add_argument("--log", default="api_test.log")
    ap.add_argument("--timeout", type=float, default=1.0)
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "production")

    out_f = open("api_server_stdout.log", "w", encoding="utf-8")
    err_f = open("api_server_stderr.log", "w", encoding="utf-8")

    # 直接啟動 api.py；若你用 Flask CLI，改成註解內那行
    # proc = subprocess.Popen([sys.executable, "-m", "flask", "run", "--host", args.host, "--port", str(args.port)],
    #                         stdout=out_f, stderr=err_f, text=True, env={**env, "FLASK_APP": args.entry})
    proc = subprocess.Popen([sys.executable, args.entry], stdout=out_f, stderr=err_f, text=True, env=env)

    url = f"http://{args.host}:{args.port}{args.endpoint}"
    method_used, status, elapsed, text = "GET", -1, -1.0, ""

    try:
        # 等待最多 25s，同時監控子行程是否提前死亡
        start = time.time()
        ok = False
        while time.time() - start < 25:
            if proc.poll() is not None:
                break
            if wait_port(args.host, args.port, timeout=0.5):
                ok = True
                break
        if not ok:
            out_f.flush(); err_f.flush()
            out_txt = Path("api_server_stdout.log").read_text(encoding="utf-8", errors="ignore")
            err_txt = Path("api_server_stderr.log").read_text(encoding="utf-8", errors="ignore")
            with open(args.log, "w", encoding="utf-8") as f:
                f.write(f"url={url}\n")
                f.write("server_start=FAIL\n")
                f.write("server_stdout_begin\n"); f.write(out_txt); f.write("\nserver_stdout_end\n")
                f.write("server_stderr_begin\n"); f.write(err_txt); f.write("\nserver_stderr_end\n")
                f.write("result=FAIL\n")
            raise SystemExit("Server didn't start")

        # 送請求（GET；若 405 改 POST）
        import time as _t, requests
        t0 = _t.perf_counter()
        try:
            r = requests.get(url, timeout=2.0)
            method_used, status, text = "GET", r.status_code, (r.text or "")[:400]
            if status == 405:
                t0 = _t.perf_counter()
                r = requests.post(url, json={"ping": "pong"}, timeout=2.0)
                method_used, status, text = "POST", r.status_code, (r.text or "")[:400]
        except requests.RequestException as e:
            status, text = -1, f"request error: {e}"
        elapsed = _t.perf_counter() - t0

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
        try: proc.terminate(); proc.wait(timeout=3)
        except Exception:
            try: proc.kill()
            except Exception: pass
        out_f.close(); err_f.close()

if __name__ == "__main__":
    main()
