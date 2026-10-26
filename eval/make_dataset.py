"""Build the labelled synthetic test set: ad-like creatives with known defects.

Photos come from scikit-image's bundled sample data (public domain / CC0, see
README); backgrounds, copy, logos and defects are generated here, so every
label is known by construction. Deterministic: same seed, same set.

    python eval/make_dataset.py --out eval/dataset --count 60
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import skimage.data as skdata

CANVASES = {"16x9": (1920, 1080), "1x1": (1080, 1080), "9x16": (1080, 1920), "4x5": (1080, 1350)}
LAYOUTS = ["photo_left", "photo_right", "photo_top", "photo_bottom", "photo_corner"]
# face boxes in the 512px base photos (YuNet on the clean photo, checked by eye)
BASES = {
    "astronaut": [178, 63, 90, 113],
    "camera": [196, 113, 61, 81],
    "chelsea": None,
    "coffee": None,
    "rocket": None,
    "product": None,
}
HEADLINES = ["SUMMER SALE 50% OFF", "NEW FLAVOUR IS HERE", "BOOK YOUR TRIP NOW", "FREE DELIVERY TODAY", "LIMITED EDITION"]
SUBLINES = ["Shop now at nova.example", "Only this weekend", "Tap to learn more", "Terms apply"]
BODY = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"


def base_photo(name):
    if name == "product":
        return product_shot()
    image = getattr(skdata, name)()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def product_shot():
    """A drawn bottle on a studio backdrop: a no-face subject with clear shape."""
    image = np.full((600, 450, 3), (215, 225, 230), np.uint8)
    cv2.ellipse(image, (225, 560), (150, 25), 0, 0, 360, (170, 180, 185), -1)
    cv2.rectangle(image, (160, 200), (290, 560), (40, 120, 60), -1)
    cv2.rectangle(image, (195, 110), (255, 200), (40, 120, 60), -1)
    cv2.rectangle(image, (190, 80), (260, 115), (30, 30, 160), -1)
    cv2.rectangle(image, (165, 300), (285, 430), (240, 240, 240), -1)
    cv2.putText(image, "NOVA", (178, 375), cv2.FONT_HERSHEY_DUPLEX, 1.1, (30, 30, 160), 2)
    for i in range(6):
        cv2.line(image, (172 + i * 4, 210), (172 + i * 4, 540), (70, 160, 90), 1)
    return image


def make_logo(variant="nova"):
    logo = np.full((160, 420, 3), 255, np.uint8)
    if variant == "nova":
        cv2.circle(logo, (80, 80), 62, (200, 60, 30), -1)
        cv2.circle(logo, (80, 80), 40, (255, 255, 255), 6)
        cv2.line(logo, (40, 120), (120, 40), (255, 255, 255), 8)
        cv2.putText(logo, "NOVA", (165, 105), cv2.FONT_HERSHEY_DUPLEX, 2.4, (200, 60, 30), 5)
        cv2.putText(logo, "drinks co.", (170, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    else:  # a different brand, used as a distractor
        cv2.rectangle(logo, (20, 20), (140, 140), (40, 160, 40), -1)
        cv2.putText(logo, "Z", (50, 120), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (255, 255, 255), 6)
        cv2.putText(logo, "ZENTA", (160, 110), cv2.FONT_HERSHEY_TRIPLEX, 2.2, (40, 160, 40), 4)
    return logo


def gradient(width, height, rng):
    top = np.array([rng.randint(20, 235) for _ in range(3)], np.float32)
    bottom = np.array([rng.randint(20, 235) for _ in range(3)], np.float32)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    return np.broadcast_to(top * (1 - ramp) + bottom * ramp, (height, width, 3)).astype(np.uint8).copy()


def put_text_line(canvas, text, x, y, max_width, font, thickness, color, max_height=None):
    """Draw one line scaled to fit max_width (and max_height); return its box [x, y, w, h]."""
    (w, h), base = cv2.getTextSize(text, font, 1.0, thickness)
    scale = max_width / w if not max_height else min(max_width / w, max_height / (h + base))
    thickness = max(1, round(thickness * scale))
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, (int(x), int(y + h)), font, scale, color, thickness, cv2.LINE_AA)
    return [int(x), int(y), w, h + base]


def regions(layout, width, height):
    """(photo region, text region) as [x, y, w, h]."""
    if layout == "photo_left":
        return [0, 0, width * 0.55, height], [width * 0.57, height * 0.2, width * 0.4, height * 0.5]
    if layout == "photo_right":
        return [width * 0.45, 0, width * 0.55, height], [width * 0.05, height * 0.2, width * 0.38, height * 0.5]
    if layout == "photo_top":
        return [0, 0, width, height * 0.6], [width * 0.1, height * 0.64, width * 0.8, height * 0.22]
    if layout == "photo_bottom":
        return [0, height * 0.4, width, height * 0.6], [width * 0.1, height * 0.1, width * 0.8, height * 0.22]
    # photo_corner: subject pushed hard into a corner, text across the middle
    return [width * 0.6, 0, width * 0.4, height * 0.4], [width * 0.08, height * 0.45, width * 0.8, height * 0.2]


def make_creative(index, rng, logo, distractor):
    canvas_name = rng.choice(list(CANVASES))
    width, height = CANVASES[canvas_name]
    layout = rng.choice(LAYOUTS)
    base_name = rng.choice(list(BASES))
    defect = rng.choices(["none", "blur_mild", "blur_severe", "dark", "text_heavy"], [0.4, 0.15, 0.12, 0.18, 0.15])[0]
    logo_mode = rng.choices(["safe", "edge", "absent", "distractor"], [0.4, 0.35, 0.15, 0.10])[0]

    canvas = gradient(width, height, rng)
    photo = base_photo(base_name)
    photo_region, text_region = regions(layout, width, height)
    scale = min(photo_region[2] * 0.9 / photo.shape[1], photo_region[3] * 0.9 / photo.shape[0])
    pw, ph = int(photo.shape[1] * scale), int(photo.shape[0] * scale)
    px = int(photo_region[0] + (photo_region[2] - pw) / 2)
    py = int(photo_region[1] + (photo_region[3] - ph) / 2)
    resized = cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if defect == "blur_mild":
        resized = cv2.GaussianBlur(resized, (0, 0), 2.5)
    if defect == "blur_severe":
        resized = cv2.GaussianBlur(resized, (0, 0), 7)
    canvas[py:py + ph, px:px + pw] = resized
    subject = [px, py, pw, ph]
    face = BASES[base_name]
    faces = [[px + face[0] * scale, py + face[1] * scale, face[2] * scale, face[3] * scale]] if face else []

    white = (255, 255, 255)
    tx, ty, tw, th = text_region
    text_boxes = [put_text_line(canvas, rng.choice(HEADLINES), tx, ty, tw, cv2.FONT_HERSHEY_DUPLEX, 3, white, th * 0.4)]
    headline = text_boxes[0]
    text_boxes.append(put_text_line(canvas, rng.choice(SUBLINES), tx, headline[1] + headline[3] * 1.4, tw * 0.7,
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, white, th * 0.25))
    if defect == "text_heavy":  # paragraphs of body copy over most of the frame
        y = height * 0.05
        while y < height * 0.9:
            text_boxes.append(put_text_line(canvas, BODY, width * 0.05, y, width * 0.9, cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 250, 250)))
            y += text_boxes[-1][3] * 1.5

    logo_box = None
    if logo_mode in ("safe", "edge", "distractor"):
        mark = logo if logo_mode != "distractor" else distractor
        lw = int(width * rng.uniform(0.16, 0.24))
        lh = int(mark.shape[0] * lw / mark.shape[1])
        corner = rng.choice(["bl", "br", "tl", "tr"])
        margin = int(0.01 * width) if logo_mode == "edge" else int(0.10 * max(width, height) * 0.5 + 0.06 * width)
        lx = margin if corner[1] == "l" else width - lw - margin
        ly = (height - lh - margin) if corner[0] == "b" else margin
        canvas[ly:ly + lh, lx:lx + lw] = cv2.resize(mark, (lw, lh), interpolation=cv2.INTER_AREA)
        if logo_mode != "distractor":
            logo_box = [lx, ly, lw, lh]

    if defect == "dark":
        canvas = (canvas.astype(np.float32) * rng.uniform(0.18, 0.3)).astype(np.uint8)

    file_name = f"c{index:03d}_{canvas_name}_{base_name}_{defect}_{logo_mode}.png"
    labels = {
        "blurry": defect in ("blur_mild", "blur_severe"),
        "severely_blurry": defect == "blur_severe",
        "dark": defect == "dark",
        "text_heavy": defect == "text_heavy",
        "logo_present": logo_box is not None,
        "face_present": bool(faces),
    }
    truth = {"faces": faces, "logo": logo_box, "headline": headline, "text": text_boxes, "subject": subject}
    meta = {"canvas": canvas_name, "layout": layout, "base": base_name, "defect": defect, "logo_mode": logo_mode}
    return canvas, {"file": file_name, "labels": labels, "truth": truth, "meta": meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/dataset")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    logo, distractor = make_logo("nova"), make_logo("zenta")
    cv2.imwrite(os.path.join(args.out, "logo_nova.png"), logo)
    items = []
    for index in range(args.count):
        image, item = make_creative(index, rng, logo, distractor)
        cv2.imwrite(os.path.join(args.out, item["file"]), image)
        items.append(item)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump({"seed": args.seed, "logo": "logo_nova.png", "items": items}, f, indent=1)
    print(f"wrote {len(items)} creatives to {args.out}")


if __name__ == "__main__":
    main()
