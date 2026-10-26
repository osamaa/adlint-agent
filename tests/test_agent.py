"""Behaviour tests: the agent's geometry, tools and decisions on tiny synthetic inputs (no network)."""

import numpy as np

from adlint import tools
from adlint.agent import placement_issues
from adlint.placements import PLACEMENTS, safe_rect
from adlint.planner import RulePlanner

STORIES = PLACEMENTS["stories_9x16"]


def elements(**kw):
    base = {"size": [1920, 1080], "faces": [], "logo": None, "text": [], "subject": None}
    base.update(kw)
    return base


def no_output_detections():
    return {"faces": {"items": []}, "text": {"boxes": []}, "logo": None}


def test_smart_crop_keeps_off_centre_face_inside_safe_zone():
    face = [1600, 300, 120, 150]
    transform, fits = tools.smart_crop(elements(faces=[face]), STORIES)
    mapped = tools.map_box(face, transform, STORIES)
    sx, sy, sw, sh = safe_rect(STORIES)
    assert fits and sx <= mapped[0] and mapped[0] + mapped[2] <= sx + sw and sy <= mapped[1] <= sy + sh


def test_centre_crop_would_cut_that_face_but_agent_check_catches_it():
    face = [1750, 300, 120, 150]
    transform = {"kind": "crop", "rect": [656, 0, 607.5, 1080]}
    issues, _ = placement_issues(elements(faces=[face]), transform, no_output_detections(), STORIES)
    assert "face_cut" in {i["code"] for i in issues}


def test_fit_into_safe_zone_keeps_whole_frame_clear_of_ui():
    transform = tools.fit_into_safe_zone(elements(), STORIES)
    mapped = tools.map_box([0, 0, 1920, 1080], transform, STORIES)
    sx, sy, sw, sh = safe_rect(STORIES)
    assert mapped[0] >= sx - 1 and mapped[1] >= sy - 1 and mapped[0] + mapped[2] <= sx + sw + 1


def test_render_produces_exact_target_size():
    image = np.zeros((1080, 1920, 3), np.uint8)
    for transform in (tools.smart_crop(elements(), STORIES)[0], tools.fit_into_safe_zone(elements(), STORIES)):
        assert tools.render(image, transform, STORIES).shape == (1920, 1080, 3)


def test_planner_escalates_when_every_reframe_was_tried():
    context = {"stage": "placement", "has_candidate": True, "aspect_match": False, "priority_differs": True,
               "candidate_kind": "fit", "issues": [{"code": "face_cut"}],
               "tried": [{"action": "smart_crop", "args": {"strategy": "all"}},
                         {"action": "smart_crop", "args": {"strategy": "priority"}},
                         {"action": "fit_into_safe_zone", "args": {}}]}
    assert RulePlanner().decide(context)["action"] == "escalate"


def test_planner_never_sharpens_severe_blur():
    context = {"stage": "source", "tried": [], "issues": [{"code": "severely_blurry"}]}
    assert RulePlanner().decide(context)["action"] == "escalate"


def test_enhance_exposure_brightens_dark_image():
    dark = np.random.default_rng(0).integers(0, 40, (200, 200, 3), dtype=np.uint8)
    assert tools.enhance_exposure(dark).mean() > dark.mean() + 30
