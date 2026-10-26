"""Perception: everything the agent can see in one image, measured with OpenCV 5.

perceive(image, logo) returns a plain JSON-able dict ("observation"):
quality (sharpness, exposure, contrast), text boxes, faces, main subject, logo.
Boxes are [x, y, w, h] in the pixel space of the image that was passed in.
"""

import os
import threading
import time

import cv2
import numpy as np

from .geometry import area, fraction_inside, intersection_area, rounded, union_box

MODELS_DIR = os.environ.get(
    "ADLINT_MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
)
FACE_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
TEXT_MODEL = os.path.join(MODELS_DIR, "text_detection_en_ppocrv3_2023may.onnx")

# Decision thresholds. Tuned on the dev set (seed 2026), then frozen; the
# reported numbers come from the held-out set (seed 2027). See TECHNICAL_REPORT.md.
THRESHOLDS = {
    "sharpness_blurry": 2.3,     # below = blurry (fixable with mild sharpening)
    "sharpness_severe": 0.8,     # below = too blurry to fix, needs a re-export
    "brightness_dark": 60.0,     # mean luma below = too dark
    "dark_pixel_share": 0.50,    # or this share of pixels under luma 40
    "contrast_low": 28.0,        # luma std below = flat / washed out
    "clipped_share": 0.30,       # share of pixels at luma >= 250 = blown out
    "text_heavy_share": 0.25,    # text boxes covering more than this = text heavy
    "logo_min_inliers": 12,      # RANSAC inliers for an ORB logo match
    "logo_template_score": 0.80, # normalised correlation for the template fallback
}

_local = threading.local()  # cv2 DNN nets are not thread-safe: one set per thread


def _face_detector():
    if not hasattr(_local, "face"):
        if os.path.exists(FACE_MODEL):
            _local.face = ("yunet", cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.7, 0.3, 50))
        else:
            cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            _local.face = ("haar", cv2.CascadeClassifier(cascade))
    return _local.face


def _text_detector():
    if not hasattr(_local, "text"):
        if os.path.exists(TEXT_MODEL):
            model = cv2.dnn.TextDetectionModel_DB(TEXT_MODEL)
            model.setBinaryThreshold(0.3)
            model.setPolygonThreshold(0.5)
            model.setMaxCandidates(200)
            model.setUnclipRatio(2.0)
            _local.text = ("db_ppocrv3", model)
        else:
            _local.text = ("mser", cv2.MSER_create())
    return _local.text


def _resize_max(image, max_side):
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------------------------------------------------------------- text
def detect_text(image):
    kind, model = _text_detector()
    if kind == "mser":
        return kind, _detect_text_mser(image, model)
    small, scale = _resize_max(image, 960)
    height, width = small.shape[:2]
    input_size = (max(32, round(width / 32) * 32), max(32, round(height / 32) * 32))
    model.setInputParams(1.0 / 255.0, input_size, (122.67891434, 116.66876762, 104.00698793))
    polygons, _ = model.detect(small)
    boxes = []
    for polygon in polygons:
        x, y, w, h = cv2.boundingRect(np.array(polygon, dtype=np.int32))
        if w * h < 30:
            continue
        boxes.append([x / scale, y / scale, w / scale, h / scale])
    return kind, boxes


def _detect_text_mser(image, mser):
    """Fallback when the DB model file is missing: MSER regions merged by morphology."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, np.uint8)
    for points in regions:
        x, y, w, h = cv2.boundingRect(points)
        if 0.1 < w / max(h, 1) < 10 and 8 < h < gray.shape[0] / 5:
            mask[y:y + h, x:x + w] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [list(cv2.boundingRect(c)) for c in contours]
    return [b for b in boxes if b[2] > 2 * b[3] and b[2] * b[3] > 200]


def text_share(boxes, width, height):
    """Share of the frame covered by text boxes (overlaps counted once)."""
    mask = np.zeros((height, width), np.uint8)
    for x, y, w, h in boxes:
        mask[int(max(0, y)):int(y + h), int(max(0, x)):int(x + w)] = 1
    return float(mask.mean())


# ---------------------------------------------------------------- faces
def detect_faces(image):
    kind, detector = _face_detector()
    small, scale = _resize_max(image, 1280)
    if kind == "haar":
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = detector.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        faces = [{"box": [v / scale for v in f], "score": 1.0} for f in found]
        return kind, faces
    height, width = small.shape[:2]
    detector.setInputSize((width, height))
    _, found = detector.detect(small)
    faces = []
    if found is not None:
        for row in found:
            faces.append({"box": [float(v) / scale for v in row[:4]], "score": float(row[-1])})
    return kind, faces


# ---------------------------------------------------------------- subject
def detect_subject(image):
    """Main subject box from spectral-residual saliency (cv2.saliency, contrib)."""
    small, scale = _resize_max(image, 400)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, saliency_map = saliency.computeSaliency(small)
    if not ok:
        return None, 0.0
    saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), 3)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None, 0.0
    weights = [saliency_map[labels == i].sum() for i in range(1, count)]
    best = int(np.argmax(weights)) + 1
    x, y, w, h = stats[best][:4]
    # keep other strong blobs that sit next to the winner (one subject, several parts)
    total = float(sum(weights))
    for i in range(1, count):
        if i != best and weights[i - 1] > 0.15 * total:
            x2, y2, w2, h2 = stats[i][:4]
            x, y, w, h = union_box([[x, y, w, h], [x2, y2, w2, h2]])
    share = weights[best - 1] / total
    return [x / scale, y / scale, w / scale, h / scale], float(share)


# ---------------------------------------------------------------- quality
def measure_quality(image, ignore_boxes=()):
    """Photo sharpness (overlay text/logo masked out) and exposure.

    Sharpness = median over textured 16x16-grid tiles of var(Laplacian) / var(pixels).
    Dividing by the tile's own contrast makes it independent of exposure, and the
    median ignores the few tiles that hold hard layout edges (a pasted photo's
    border, a button). None = nothing left to judge (e.g. a wall of text).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small, scale = _resize_max(gray, 720)
    laplacian = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
    height, width = small.shape
    ignore = [[(b[0] - 0.3 * b[3]) * scale, (b[1] - 0.3 * b[3]) * scale, (b[2] + 0.6 * b[3]) * scale, 1.6 * b[3] * scale]
              for b in ignore_boxes]
    ratios = []
    tiles = 16
    for row in range(tiles):
        for col in range(tiles):
            tile = [col * width / tiles, row * height / tiles, width / tiles, height / tiles]
            if any(intersection_area(tile, b) > 0.05 * area(tile) for b in ignore):
                continue
            x, y = int(tile[0]), int(tile[1])
            patch = laplacian[y:int(y + tile[3]), x:int(x + tile[2])]
            source = small[y:int(y + tile[3]), x:int(x + tile[2])]
            spread = float(source.std()) if source.size else 0.0
            if spread > 4:  # flat tiles hold no detail to judge
                ratios.append(float(patch.var()) / spread ** 2)
    sharpness = round(float(np.median(ratios)), 3) if ratios else None
    return {
        "sharpness": sharpness,
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "dark_share": round(float((gray < 40).mean()), 3),
        "clipped_share": round(float((gray >= 250).mean()), 3),
    }


def quality_flags(quality):
    t = THRESHOLDS
    return {
        "blurry": quality["sharpness"] is not None and quality["sharpness"] < t["sharpness_blurry"],
        "severely_blurry": quality["sharpness"] is not None and quality["sharpness"] < t["sharpness_severe"],
        "dark": quality["brightness"] < t["brightness_dark"] or quality["dark_share"] > t["dark_pixel_share"],
        "low_contrast": quality["contrast"] < t["contrast_low"],
        "overexposed": quality["clipped_share"] > t["clipped_share"],
    }


# ---------------------------------------------------------------- logo
def find_logo(image, logo):
    """Locate the brand logo: ORB features + RANSAC homography, template fallback."""
    if logo is None:
        return None
    if min(image.shape[:2]) < 32:  # too small to hold a recognisable logo (and ORB's pyramid needs room)
        return {"found": False, "box": None, "method": None, "score": 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    result = _find_logo_orb(gray, logo_gray)
    if result is None:
        result = _find_logo_template(gray, logo_gray)
    if result is None:
        return {"found": False, "box": None, "method": None, "score": 0.0}
    return result


def _find_logo_orb(gray, logo_gray):
    orb = cv2.ORB_create(nfeatures=3000)
    kp_img, des_img = orb.detectAndCompute(gray, None)
    # reference features from several logo sizes, mapped back to full-size logo
    # coordinates: one size alone stops matching once the logo appears much smaller
    kp_logo, des_logo = [], []
    for scale in (1.0, 0.6, 0.35):
        resized = cv2.resize(logo_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else logo_gray
        keypoints, descriptors = cv2.ORB_create(nfeatures=500).detectAndCompute(resized, None)
        if descriptors is not None:
            kp_logo += [cv2.KeyPoint(k.pt[0] / scale, k.pt[1] / scale, k.size / scale) for k in keypoints]
            des_logo.append(descriptors)
    des_logo = np.vstack(des_logo) if des_logo else None
    if des_logo is None or des_img is None or len(kp_logo) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_logo, des_img, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < THRESHOLDS["logo_min_inliers"]:
        return None
    src = np.float32([kp_logo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None:
        return None
    inliers = int(inlier_mask.sum())
    if inliers < THRESHOLDS["logo_min_inliers"]:
        return None
    h, w = logo_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, homography)
    if not cv2.isContourConvex(quad.astype(np.int32)) or cv2.contourArea(quad) < 100:
        return None
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    # a real match looks like the logo: about the same shape, inside the frame
    aspect_change = (bw / bh) / (w / h)
    inside = fraction_inside([x, y, bw, bh], [0, 0, gray.shape[1], gray.shape[0]])
    if not 0.7 < aspect_change < 1.4 or inside < 0.9:
        return None
    return {"found": True, "box": [x, y, bw, bh], "method": "orb_homography", "score": float(inliers)}


def _find_logo_template(gray, logo_gray):
    """Multi-scale normalised cross-correlation, for flat logos with few corners (searched at <= 640 px wide)."""
    shrink = min(1.0, 640 / gray.shape[1])
    if shrink < 1:
        gray = cv2.resize(gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    best = (0.0, None)
    for target_width in np.linspace(0.04, 0.5, 24) * gray.shape[1]:
        scale = target_width / logo_gray.shape[1]
        w, h = int(logo_gray.shape[1] * scale), int(logo_gray.shape[0] * scale)
        if w < 12 or h < 12 or w >= gray.shape[1] or h >= gray.shape[0]:
            continue
        template = cv2.resize(logo_gray, (w, h), interpolation=cv2.INTER_AREA)
        if template.std() < 5:
            continue
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(scores)
        if max_val > best[0]:
            best = (max_val, [max_loc[0], max_loc[1], w, h])
    if best[0] < THRESHOLDS["logo_template_score"]:
        return None
    box = [v / shrink for v in best[1]]
    return {"found": True, "box": box, "method": "template", "score": float(best[0])}


# ---------------------------------------------------------------- all together
def perceive(image, logo=None):
    started = time.perf_counter()
    height, width = image.shape[:2]
    text_kind, text_boxes = detect_text(image)
    face_kind, faces = detect_faces(image)
    subject, subject_share = detect_subject(image)
    logo_result = find_logo(image, logo)
    if logo_result and logo_result["found"]:  # the logo's own lettering is not ad copy
        text_boxes = [b for b in text_boxes if fraction_inside(b, logo_result["box"]) < 0.6]
    ignore = list(text_boxes) + ([logo_result["box"]] if logo_result and logo_result["found"] else [])
    quality = measure_quality(image, ignore)
    return {
        "size": [width, height],
        "aspect": round(width / height, 4),
        "quality": quality,
        "flags": quality_flags(quality),
        "text": {
            "detector": text_kind,
            "boxes": [rounded(b) for b in text_boxes],
            "share": round(text_share(text_boxes, width, height), 3),
        },
        "faces": {"detector": face_kind, "items": [{"box": rounded(f["box"]), "score": round(f["score"], 3)} for f in faces]},
        "subject": {"box": rounded(subject) if subject else None, "saliency_share": round(subject_share, 3)},
        "logo": logo_result,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }
