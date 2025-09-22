# tests/run_text_test.py
import argparse, subprocess, sys, time, shlex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="display.py")
    ap.add_argument("--log", default="text_test.log")
    ap.add_argument("--min_lines", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, args.entry],
                              capture_output=True, text=True, timeout=args.timeout)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        lines = [ln for ln in out.splitlines() if ln.strip() != ""]
        passed = (elapsed < args.timeout) and (len(lines) >= args.min_lines) and (proc.returncode == 0)
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        out = (e.stdout or "") + (e.stderr or "")
        lines = [ln for ln in out.splitlines() if ln.strip() != ""]
        passed = False

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"elapsed_sec={elapsed:.3f}\n")
        f.write(f"lines={len(lines)}\n")
        f.write(f"criteria=elapsed<{args.timeout}s && lines>={args.min_lines}\n")
        f.write("output_begin\n")
        f.write(out)
        f.write("\noutput_end\n")
        f.write(f"result={'PASS' if passed else 'FAIL'}\n")

    assert passed, f"TEXT test failed: elapsed={elapsed:.3f}s, lines={len(lines)}"

if __name__ == "__main__":
    main()
