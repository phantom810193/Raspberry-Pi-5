# tests/run_api_test.py
import argparse, os, time, subprocess, socket, sys
from contextlib import closing
from pathlib import Path
import requests

def wait_port(host, port, timeout=6.0):
    start = time.time()
    while time.time() - start < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(0.2)
    return False

def start_proc(cmd, env, out_path, err_path):
    out_f = open(out_path, "w", encoding="utf-8")
    err_f = open(err_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True, env=env)
    return proc, out_f, err_f

def stop_proc(proc, out_f, err_f):
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired: proc.kill()
    finally:
        out_f.close(); err_f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="api.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--endpoint", default="/detect_face")
    ap.add_argument("--log", default="api_test.log")
    ap.add_argument("--timeout", type=float, default=1.0)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}{args.endpoint}"
    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "production")

    # 方案 A：python api.py
    proc, out_f, err_f = start_proc(
        [sys.executable, args.entry],
        env,
        "api_server_stdout.log",
        "api_server_stderr.log",
    )
    ok = wait_port(args.host, args.port, timeout=6.0)

    # 若失敗，換 方案 B：Flask CLI（常見於檔案沒有 if __name__ == '__main__'）
    if not ok:
        stop_proc(proc, out_f, err_f)
        env2 = env.copy()
        env2["FLASK_APP"] = args.entry
        env2["FLASK_RUN_FROM_CLI"] = "true"
        proc, out_f, err_f = start_proc(
            [sys.executable, "-m", "flask", "run", "--host", args.host, "--port", str(args.port)],
            env2,
            "api_server_stdout.log",
            "api_server_stderr.log",
        )
        ok = wait_port(args.host, args.port, timeout=10.0)

    if not ok:
        # 讀取器輸出，寫到主 log
        out_f.flush(); err_f.flush()
        out_txt = Path("api_server_stdout.log").read_text(encoding="utf-8", errors="ignore")
        err_txt = Path("api_server_stderr.log").read_text(encoding="utf-8", errors="ignore")
        with open(args.log, "w", encoding="utf-8") as f:
            f.write(f"url={url}\n")
            f.write("server_start=FAIL\n")
            f.write("server_stdout_begin\n"); f.write(out_txt); f.write("\nserver_stdout_end\n")
            f.write("server_stderr_begin\n"); f.write(err_txt); f.write("\nserver_stderr_end\n")
            f.write("result=FAIL\n")
        stop_proc(proc, out_f, err_f)
        raise SystemExit("Server didn't start")

    # 打一次請求（GET→405 改 POST）
    t0 = time.perf_counter()
    method, status, text = "GET", -1, ""
    try:
        r = requests.get(url, timeout=2.0)
        status, text = r.status_code, (r.text or "")[:400]
        if status == 405:
            t0 = time.perf_counter()
            r = requests.post(url, json={"ping": "pong"}, timeout=2.0)
            method, status, text = "POST", r.status_code, (r.text or "")[:400]
    except requests.RequestException as e:
        status, text = -1, f"request error: {e}"
    elapsed = time.perf_counter() - t0

    passed = (status in (200, 201)) and (elapsed < args.timeout)
    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"url={url}\n")
        f.write(f"method={method}\n")
        f.write(f"status={status}\n")
        f.write(f"elapsed_sec={elapsed:.3f}\n")
        f.write(f"body_preview={text}\n")
        f.write(f"criteria=http_2xx && elapsed<{args.timeout}s\n")
        f.write(f"result={'PASS' if passed else 'FAIL'}\n")

    stop_proc(proc, out_f, err_f)
    assert passed, f"API test failed: status={status}, elapsed={elapsed:.3f}s"

if __name__ == "__main__":
    main()
