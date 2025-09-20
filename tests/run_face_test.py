# tests/run_face_test.py
import argparse, io, math, urllib.request, time
from pathlib import Path
from typing import Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
import face_recognition as fr

SAMPLE_URL = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"

def load_image_from_url(url: str) -> Image.Image:
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

def _prepare_image(im: Image.Image, max_w=1600) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > max_w:
        h = int(im.height * (max_w / im.width))
        im = im.resize((max_w, h))
    return im

def _largest_box(boxes):
    if not boxes:
        return None
    areas = [max(0, (b - t)) * max(0, (r - l)) for (t, r, b, l) in boxes]
    return boxes[int(np.argmax(areas))]

def encode_one(im: Image.Image, debug_path: Optional[str] = None) -> np.ndarray:
    im = _prepare_image(im)
    arr = np.array(im)
    boxes = fr.face_locations(arr, number_of_times_to_upsample=1, model="hog")
    if not boxes:
        boxes = fr.face_locations(arr, number_of_times_to_upsample=2, model="hog")
    if not boxes:
        try:
            boxes = fr.face_locations(arr, number_of_times_to_upsample=1, model="cnn")
        except Exception:
            boxes = []
    if not boxes:
        raise RuntimeError("No face encoding found")
    box = _largest_box(boxes)
    enc = fr.face_encodings(arr, known_face_locations=[box])[0]
    if debug_path:
        dbg_im = im.copy()
        draw = ImageDraw.Draw(dbg_im)
        t, r, b, l = box
        draw.rectangle(((l, t), (r, b)), outline=(255, 0, 0), width=4)
        dbg_im.save(debug_path, format="JPEG", quality=90)
    return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="id_test.log")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--target_pass_ratio", type=float, default=0.8)
    ap.add_argument("--input-dir", default=None, help="Directory of your own photos (>=5)")
    ap.add_argument("--save-dir", default=None, help="Directory to save generated sample variants OR debug boxed images")
    args = ap.parse_args()

    t0 = time.time()

    names = []
    if args.input_dir:
        p = Path(args.input_dir)
        if not p.exists():
            raise RuntimeError(f"Input dir not found: {p}")
        exts = (".jpg", ".jpeg", ".png")
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        if len(files) < 5:
            raise RuntimeError(f"Need >=5 images in {p}, found {len(files)}")
        names = [files[i].name for i in range(5)]
        debug_dir = Path(args.save_dir) if args.save_dir else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
        imgs = [Image.open(f).convert("RGB") for f in files[:5]]
        encs = []
        for i, im in enumerate(imgs):
            dbg = str(debug_dir / f"input_{i+1}_boxed.jpg") if debug_dir else None
            try:
                encs.append(encode_one(im, debug_path=dbg))
            except Exception as e:
                raise RuntimeError(f"No face found in file: {files[i].name}") from e
    else:
        base = load_image_from_url(SAMPLE_URL)
        variants = make_variants(base)
        names = ["variant_0_base.jpg","variant_1_rotate.jpg","variant_2_brightness.jpg","variant_3_crop.jpg","variant_4_blur.jpg"]
        outdir = Path(args.save_dir) if args.save_dir else None
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            for im, name in zip(variants, names):
                im.save(outdir / name, format="JPEG", quality=95)
        encs = [encode_one(im) for im in variants]

    ref = encs[0]
    others = encs[1:]
    dists = fr.face_distance(others, ref)
    matches_bools = [bool(float(d) < args.threshold) for d in dists]
    matches = int(sum(matches_bools))
    ratio = matches / len(others)
    required = math.ceil(args.target_pass_ratio * len(others))
    passed = matches >= required

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"threshold={args.threshold}\n")
        f.write(f"required_matches={required} / {len(others)} (>= {args.target_pass_ratio*100:.0f}%)\n")
        for fname, d, ok in zip(names[1:], dists, matches_bools):
            f.write(f"{fname}: distance={float(d):.4f}, match={ok}\n")
        f.write(f"total_matches={matches}\n")
        f.write(f"pass_ratio={ratio:.2f}\n")
        f.write(f"result={'PASS' if passed else 'FAIL'}\n")
        f.write(f"elapsed_sec={time.time()-t0:.2f}\n")

    with open("id_test_files.txt", "w", encoding="utf-8") as f:
        f.write("index,filename,distance,match\n")
        for i, (fname, d, ok) in enumerate(zip(names[1:], dists, matches_bools), start=1):
            f.write(f"{i},{fname},{float(d):.6f},{int(ok)}\n")

    assert passed, f"Face-ID test did not reach {args.target_pass_ratio*100:.0f}% (got {ratio*100:.0f}%)"

if __name__ == "__main__":
    main()
