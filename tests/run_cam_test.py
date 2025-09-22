## tests/run_cam_test.py
import argparse, time, numpy as np
import cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=5.0)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--log', type=str, default='cam.log')
    args = p.parse_args()

    W, H = args.width, args.height
    start = time.time()
    frames = 0

    while time.time() - start < args.seconds:
        frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _ = cv2.Canny(gray, 50, 150)
        frames += 1

    elapsed = time.time() - start
    fps = frames / elapsed if elapsed > 0 else 0.0

    with open(args.log, 'w', encoding='utf-8') as f:
        f.write(f'seconds={elapsed:.3f}\n')
        f.write(f'frames={frames}\n')
        f.write(f'fps={fps:.2f}\n')
        f.write('criteria=FPS>10\n')
        f.write(f'result={"PASS" if fps>10 else "FAIL"}\n')

    assert fps > 10.0, f'FPS too low: {fps:.2f}'

if __name__ == '__main__':
    main()
