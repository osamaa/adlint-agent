"""Action tools the agent can call. Each one is a plain OpenCV operation.

Reframing tools return a *transform* (how source pixels land in the output),
so the agent can map what it saw in the source into the output and check it.
  crop: {"kind": "crop", "rect": [x, y, w, h]}            source rect -> full output
  fit:  {"kind": "fit", "scale": s, "offset": [ox, oy]}    whole source, scaled, on a blurred backdrop
"""

import math

import cv2
import numpy as np

from .geometry import expand, union_box
from .placements import safe_rect


# ---------------------------------------------------------------- quality fixes
def enhance_exposure(image):
    """Lift dark or flat images: gamma towards mid-grey, then CLAHE on lightness."""
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    out = image
    if 5 < gray_mean < 100:
        gamma = math.log(115 / 255) / math.log(gray_mean / 255)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
        out = cv2.LUT(out, table)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(image, amount=1.0, sigma=2.0):
    """Unsharp mask. Only meant for mild softness; it cannot recover lost detail."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


# ---------------------------------------------------------------- reframing
def keep_box(elements, strategy):
    """What must survive the reframe.

    all      = faces (with head room) + logo + text + main subject
    priority = faces + logo only; text and the saliency subject may be cut
    """
    width, height = elements["size"]
    faces = [expand(f, 1.5, width, height) for f in elements["faces"]]
    boxes = faces + ([elements["logo"]] if elements["logo"] else [])
    if strategy == "all":
        boxes += elements["text"] + ([elements["subject"]] if elements["subject"] else [])
    return union_box(boxes)


def smart_crop(elements, placement, strategy="all"):
    """Largest crop at the target aspect that puts the keep-box inside the safe zone.

    Returns (transform, fits) where fits says whether the keep-box can sit
    fully inside the safe zone at all for this crop size.
    """
    width, height = elements["size"]
    target_w, target_h = placement["size"]
    aspect = target_w / target_h
    crop_w = min(width, height * aspect)
    crop_h = crop_w / aspect
    top, bottom, left, right = placement["unsafe"]
    safe_w, safe_h = 1 - left - right, 1 - top - bottom

    keep = keep_box(elements, strategy) or [width / 2, height / 2, 0, 0]
    fits = keep[2] <= crop_w * safe_w + 1 and keep[3] <= crop_h * safe_h + 1
    x = _place_axis(keep[0], keep[2], crop_w, left, right, width)
    y = _place_axis(keep[1], keep[3], crop_h, top, bottom, height)
    return {"kind": "crop", "rect": [round(x, 1), round(y, 1), round(crop_w, 1), round(crop_h, 1)]}, bool(fits)


def _place_axis(start, size, window, near_margin, far_margin, limit):
    """1-D placement: centre the keep span in the window's safe band, stay in the image."""
    safe_centre = (near_margin + (1 - near_margin - far_margin) / 2) * window
    position = start + size / 2 - safe_centre
    lowest = start + size - window * (1 - far_margin)   # span's far edge inside the band
    highest = start - window * near_margin              # span's near edge inside the band
    if lowest <= highest:
        position = min(max(position, lowest), highest)
    return min(max(position, 0.0), limit - window)


def fit_into_safe_zone(elements, placement):
    """Shrink the whole source into the safe zone; fill the rest with a blurred copy."""
    width, height = elements["size"]
    sx, sy, sw, sh = safe_rect(placement)
    scale = min(sw / width, sh / height)
    offset_x = sx + (sw - width * scale) / 2
    offset_y = sy + (sh - height * scale) / 2
    return {"kind": "fit", "scale": round(scale, 5), "offset": [round(offset_x, 1), round(offset_y, 1)]}


def identity_resize(elements, placement):
    width, height = elements["size"]
    return {"kind": "crop", "rect": [0, 0, width, height]}


def render(image, transform, placement):
    target_w, target_h = placement["size"]
    if transform["kind"] == "crop":
        x, y, w, h = transform["rect"]
        x0, y0 = int(round(x)), int(round(y))
        region = image[y0:y0 + int(round(h)), x0:x0 + int(round(w))]
        interpolation = cv2.INTER_AREA if region.shape[1] > target_w else cv2.INTER_CUBIC
        return cv2.resize(region, (target_w, target_h), interpolation=interpolation)

    height, width = image.shape[:2]
    cover = max(target_w / width, target_h / height)
    backdrop = cv2.resize(image, (math.ceil(width * cover), math.ceil(height * cover)), interpolation=cv2.INTER_AREA)
    bx, by = (backdrop.shape[1] - target_w) // 2, (backdrop.shape[0] - target_h) // 2
    backdrop = backdrop[by:by + target_h, bx:bx + target_w]
    # heavy blur on an 1/8 copy, then upscale: same look, ~50x cheaper (matters for video)
    tiny = cv2.resize(backdrop, (max(1, target_w // 8), max(1, target_h // 8)), interpolation=cv2.INTER_AREA)
    tiny = cv2.GaussianBlur(tiny, (0, 0), max(target_w, target_h) / 320)
    backdrop = cv2.resize(tiny, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    out = (backdrop * 0.55).astype(np.uint8)
    scale = transform["scale"]
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ox, oy = int(round(transform["offset"][0])), int(round(transform["offset"][1]))
    out[oy:oy + new_h, ox:ox + new_w] = small[: target_h - oy, : target_w - ox]
    return out


def map_box(box, transform, placement):
    """Where a source box lands in the output frame (not clipped)."""
    if box is None:
        return None
    if transform["kind"] == "crop":
        x, y, w, h = transform["rect"]
        sx, sy = placement["size"][0] / w, placement["size"][1] / h
        return [(box[0] - x) * sx, (box[1] - y) * sy, box[2] * sx, box[3] * sy]
    scale = transform["scale"]
    ox, oy = transform["offset"]
    return [box[0] * scale + ox, box[1] * scale + oy, box[2] * scale, box[3] * scale]
