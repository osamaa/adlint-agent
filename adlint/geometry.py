"""Axis-aligned box helpers. A box is [x, y, w, h] in pixels."""


def area(box):
    return max(0.0, box[2]) * max(0.0, box[3])


def intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def intersection_area(a, b):
    inter = intersection(a, b)
    return area(inter) if inter else 0.0


def union_box(boxes):
    boxes = [b for b in boxes if b is not None and area(b) > 0]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def iou(a, b):
    inter = intersection_area(a, b)
    total = area(a) + area(b) - inter
    return inter / total if total else 0.0


def fraction_inside(box, region):
    """Share of the box's area that lies inside region (0..1)."""
    box_area = area(box)
    return intersection_area(box, region) / box_area if box_area else 0.0


def expand(box, factor, limit_w=None, limit_h=None):
    """Grow a box around its centre; optionally clip to an image size."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    w, h = box[2] * factor, box[3] * factor
    out = [cx - w / 2, cy - h / 2, w, h]
    if limit_w is not None:
        out = intersection(out, [0, 0, limit_w, limit_h]) or out
    return out


def rounded(box):
    return [round(float(v), 1) for v in box]
