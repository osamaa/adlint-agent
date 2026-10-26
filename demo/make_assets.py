"""Showcase creatives for the demo video (all generated; photos are skimage CC0 / public domain)."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.make_dataset import base_photo, make_logo, put_text_line  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WHITE = (255, 255, 255)


def backdrop(width, height, top, bottom):
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    image = np.array(top, np.float32) * (1 - ramp) + np.array(bottom, np.float32) * ramp
    return np.broadcast_to(image, (height, width, 3)).astype(np.uint8).copy()


def paste(canvas, image, x, y, width):
    height = round(image.shape[0] * width / image.shape[1])
    canvas[y:y + height, x:x + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def wide_hero(logo):
    """16:9 YouTube master: face right, copy left, logo hugging the bottom-left edge."""
    canvas = backdrop(1920, 1080, (120, 60, 20), (200, 120, 40))
    paste(canvas, base_photo("astronaut"), 1060, 120, 800)
    put_text_line(canvas, "BOOK YOUR TRIP NOW", 70, 330, 900, cv2.FONT_HERSHEY_DUPLEX, 3, WHITE)
    put_text_line(canvas, "Orbit weekends from $999", 70, 470, 640, cv2.FONT_HERSHEY_SIMPLEX, 2, WHITE)
    paste(canvas, logo, 20, 920, 380)
    return canvas


def dark_square(logo):
    """1:1 feed post shot too dark."""
    canvas = backdrop(1080, 1080, (60, 90, 40), (20, 40, 20))
    paste(canvas, base_photo("product"), 300, 60, 480)
    put_text_line(canvas, "NEW FLAVOUR IS HERE", 120, 740, 840, cv2.FONT_HERSHEY_DUPLEX, 3, WHITE)
    paste(canvas, logo, 690, 930, 330)
    return (canvas * 0.3).astype(np.uint8)


def story_cta_bottom(logo):
    """9:16 story whose call-to-action sits where Reels puts its caption and buttons."""
    canvas = backdrop(1080, 1920, (150, 40, 120), (40, 20, 80))
    paste(canvas, base_photo("coffee"), 90, 360, 900)
    paste(canvas, logo, 40, 40, 360)
    put_text_line(canvas, "FREE DELIVERY TODAY", 90, 1420, 900, cv2.FONT_HERSHEY_DUPLEX, 3, WHITE)
    put_text_line(canvas, "Tap to order", 90, 1660, 500, cv2.FONT_HERSHEY_SIMPLEX, 2, WHITE)
    return canvas


def blurry_text_wall(logo):
    """Soft photo, wall of small print: the agent should refuse to sign this off."""
    canvas = backdrop(1080, 1350, (30, 30, 30), (90, 90, 90))
    paste(canvas, cv2.GaussianBlur(base_photo("chelsea"), (0, 0), 6), 0, 300, 1080)
    y = 40
    while y < 1250:
        box = put_text_line(canvas, "Offer valid for new customers only, see terms and conditions", 40, y, 1000,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240))
        y += int(box[3] * 1.6)
    return canvas


def promo_video(logo, path, fps=24):
    """8 s 16:9 promo: 4 shots, a dark one and a blurry end card; logo only appears late."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    astronaut, coffee, product, cat = (base_photo(n) for n in ("astronaut", "coffee", "product", "chelsea"))
    for i in range(int(2.5 * fps)):  # shot 1: slow push-in on a face, headline left
        canvas = backdrop(1920, 1080, (120, 60, 20), (200, 120, 40))
        paste(canvas, astronaut, 1100 - i * 2, 140, 760 + i * 3)
        put_text_line(canvas, "BOOK YOUR TRIP NOW", 70, 380, 900, cv2.FONT_HERSHEY_DUPLEX, 3, WHITE)
        writer.write(canvas)
    for i in range(int(2 * fps)):  # shot 2: underexposed coffee
        canvas = backdrop(1920, 1080, (20, 30, 40), (10, 10, 20))
        paste(canvas, coffee, 560 + i, 140, 800)
        writer.write((canvas * 0.35).astype(np.uint8))
    for i in range(int(2 * fps)):  # shot 3: product + logo near the edge
        canvas = backdrop(1920, 1080, (210, 220, 225), (170, 180, 190))
        paste(canvas, product, 760, 60, 450)
        paste(canvas, logo, 30, 900, 420)
        writer.write(canvas)
    for i in range(int(1.5 * fps)):  # shot 4: blurry end card
        canvas = backdrop(1920, 1080, (40, 40, 40), (80, 80, 80))
        paste(canvas, cv2.GaussianBlur(cat, (0, 0), 5), 360, 40, 1200)
        writer.write(canvas)
    writer.release()


def main():
    os.makedirs(OUT, exist_ok=True)
    logo = make_logo("nova")
    cv2.imwrite(os.path.join(OUT, "logo_nova.png"), logo)
    for name, build in (("1_wide_hero.jpg", wide_hero), ("2_dark_square.jpg", dark_square),
                        ("3_story_cta_bottom.jpg", story_cta_bottom), ("4_blurry_text_wall.jpg", blurry_text_wall)):
        cv2.imwrite(os.path.join(OUT, name), build(logo), [cv2.IMWRITE_JPEG_QUALITY, 93])
    promo_video(logo, os.path.join(OUT, "5_promo.mp4"))
    print("demo assets in", OUT)


if __name__ == "__main__":
    main()
