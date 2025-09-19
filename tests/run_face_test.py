## tests/run_face_test.py
import argparse, io, math, urllib.request, time, os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import face_recognition as fr

SAMPLE_URL = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"

def load_image_from_url(url: str):
    data = urllib.request.urlopen(url, timeout=30).read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def make_variants(img: Image.Image):
    variants = []
    variants.append(img.copy())
    variants.append(img.rotate(2, expand=True))
    variants.append(ImageEnhance.Brightness(img).enhance(1.12))
    w, h = img.size
    crop = img.crop((int(0.03*w), int(0.03*h), int(0.97*w), int(0.97*h))).resize((w, h))
    variants.append(crop)
    variants.append(img.filter(ImageFilter.GaussianBlur(radius=0.7)))
    return variants

def encode_one(im: Image.Image):
    arr = np.array(im)
    boxes = fr.face_locations(arr, model="hog")
    enc = fr.face_encodings(arr, known_face_locations=boxes)
    if not enc:
        raise RuntimeError("No face encoding found")
    return enc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="id_test.log")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--target_pass_ratio", type=float, default=0.8)
    ap.add_argument("--input-dir", default=None, help="Directory of your own photos (>=5)")
    ap.add_argument("--save-dir", default=None, help="Directory to save generated sample variants")
    args = ap.parse_args()

    t0 = time.time()

    if args.input_dir:
        p = Path(args.input_dir)
        if not p.exists():
            raise RuntimeError(f"Input dir not found: {p}")
        exts = (".jpg", ".jpeg", ".png")
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        if len(files) < 5:
            raise RuntimeError(f"Need >=5 images in {p}, found {len(files)}")
        imgs = [Image.open(f).convert("RGB") for f in files[:5]]
        encs = [encode_one(im) for im in imgs]
    else:
        base = load_image_from_url(SAMPLE_URL)
        variants = make_variants(base)
        if args.save_dir:
            outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
            names = ["variant_0_base.jpg","variant_1_rotate.jpg","variant_2_brightness.jpg","variant_3_crop.jpg","variant_4_blur.jpg"]
            for im, name in zip(variants, names):
                im.save(outdir / name, format="JPEG", quality=95)
        encs = [encode_one(im) for im in variants]

    ref = encs[0]
    others = encs[1:]
    dists = fr.face_distance(others, ref)
    matches = int(sum(float(d) < args.threshold for d in dists))
    ratio = matches / len(others)
    required = math.ceil(args.target_pass_ratio * len(others))
    passed = matches >= required

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"threshold={args.threshold}\n")
        f.write(f"required_matches={required} / {len(others)} (>= {args.target_pass_ratio*100:.0f}%)\n")
        for i, d in enumerate(dists, 1):
            f.write(f"image_{i}_distance={d:.4f}, match={d < args.threshold}\n")
        f.write(f"total_matches={matches}\n")
        f.write(f"pass_ratio={ratio:.2f}\n")
        f.write(f"result={'PASS' if passed else 'FAIL'}\n")
        f.write(f"elapsed_sec={time.time()-t0:.2f}\n")

    assert passed, f"Face-ID test did not reach {args.target_pass_ratio*100:.0f}% (got {ratio*100:.0f}%)"

if __name__ == "__main__":
    main()
