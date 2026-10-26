"""Target placements.

Each placement is an output size plus the zones that the platform's own UI covers
(profile bar, caption, buttons, progress bar). Zones are fractions of the frame:
(top, bottom, left, right). The numbers follow the public safe-zone guides of the
platforms, rounded; they are approximations and easy to edit here.
"""

PLACEMENTS = {
    "stories_9x16": {
        "label": "Stories 9:16 (1080x1920)",
        "size": (1080, 1920),
        "unsafe": (0.14, 0.20, 0.06, 0.06),
        "min_text_px": 24,
    },
    "reels_9x16": {
        "label": "Reels / TikTok 9:16 (1080x1920)",
        "size": (1080, 1920),
        "unsafe": (0.14, 0.35, 0.06, 0.16),
        "min_text_px": 24,
    },
    "feed_1x1": {
        "label": "Feed 1:1 (1080x1080)",
        "size": (1080, 1080),
        "unsafe": (0.03, 0.03, 0.03, 0.03),
        "min_text_px": 20,
    },
    "feed_4x5": {
        "label": "Feed 4:5 (1080x1350)",
        "size": (1080, 1350),
        "unsafe": (0.03, 0.03, 0.03, 0.03),
        "min_text_px": 20,
    },
    "youtube_16x9": {
        "label": "YouTube in-stream 16:9 (1920x1080)",
        "size": (1920, 1080),
        "unsafe": (0.08, 0.18, 0.03, 0.03),
        "min_text_px": 24,
    },
    "display_300x250": {
        "label": "Display MPU 300x250",
        "size": (300, 250),
        "unsafe": (0.04, 0.04, 0.04, 0.04),
        "min_text_px": 9,
    },
}

DEFAULT_PLACEMENTS = ["stories_9x16", "feed_1x1", "youtube_16x9", "display_300x250"]


def unsafe_rects(placement):
    """UI-covered zones as (x, y, w, h) pixel rects in the output frame."""
    width, height = placement["size"]
    top, bottom, left, right = placement["unsafe"]
    rects = []
    if top:
        rects.append((0, 0, width, round(top * height)))
    if bottom:
        rects.append((0, round((1 - bottom) * height), width, round(bottom * height)))
    if left:
        rects.append((0, 0, round(left * width), height))
    if right:
        rects.append((round((1 - right) * width), 0, round(right * width), height))
    return rects


def safe_rect(placement):
    """The area no platform UI covers, as an (x, y, w, h) pixel rect."""
    width, height = placement["size"]
    top, bottom, left, right = placement["unsafe"]
    x, y = left * width, top * height
    return (x, y, width * (1 - left - right), height * (1 - top - bottom))
