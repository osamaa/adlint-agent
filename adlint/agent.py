"""The agent loop: perceive -> decide -> act -> perceive again, bounded, fully traced.

Stage 1 (source): fix the master creative's quality (exposure, mild softness)
or flag what cannot be fixed (heavy blur, text-heavy, missing logo).
Stage 2 (per placement): reframe for each output size, then re-run perception
on the rendered output and check it against that placement's safe zones.
"""

import time

import cv2

from . import tools
from .geometry import expand, fraction_inside, intersection, rounded
from .perception import THRESHOLDS, find_logo, perceive
from .placements import PLACEMENTS, safe_rect

MAX_SOURCE_STEPS = 4
MAX_PLACEMENT_STEPS = 6
ASPECT_TOLERANCE = 0.02

SEVERITY = {
    "face_cut": 5, "logo_cut": 5, "logo_not_redetected": 3, "face_in_ui_zone": 3,
    "logo_in_ui_zone": 3, "text_cut": 3, "text_in_ui_zone": 2, "text_too_small": 2, "subject_cut": 2,
}


# ---------------------------------------------------------------- issue checks
def source_issues(observation, logo_given):
    flags = observation["flags"]
    issues = []
    if flags["severely_blurry"]:
        issues.append({"code": "severely_blurry", "detail": f"sharpness {observation['quality']['sharpness']}"})
    elif flags["blurry"]:
        issues.append({"code": "blurry", "detail": f"sharpness {observation['quality']['sharpness']}"})
    for code in ("dark", "low_contrast", "overexposed"):
        if flags[code]:
            issues.append({"code": code, "detail": f"brightness {observation['quality']['brightness']}, contrast {observation['quality']['contrast']}"})
    if observation["text"]["share"] > THRESHOLDS["text_heavy_share"]:
        issues.append({"code": "text_heavy", "detail": f"text covers {observation['text']['share']:.0%} of the frame"})
    if logo_given and not observation["logo"]["found"]:
        issues.append({"code": "logo_missing", "detail": "supplied logo not found in the creative"})
    return issues


def elements_of(observation):
    """The things that must survive a reframe, from the source observation."""
    logo = observation["logo"]
    return {
        "size": observation["size"],
        "faces": [f["box"] for f in observation["faces"]["items"]],
        "logo": logo["box"] if logo and logo["found"] else None,
        "text": observation["text"]["boxes"],
        "subject": observation["subject"]["box"],
    }


def placement_issues(elements, transform, output_observation, placement):
    """Check a rendered candidate: what got cut, what sits under platform UI, what became unreadable."""
    width, height = placement["size"]
    frame = [0, 0, width, height]
    safe = safe_rect(placement)
    issues, measures = [], {"faces": [], "logo": None, "text": []}

    def check(kind, box, keep_min, ui_max):
        mapped = tools.map_box(box, transform, placement)
        kept = fraction_inside(mapped, frame)
        visible = intersection(mapped, frame)
        under_ui = 1 - fraction_inside(visible, safe) if visible else 0.0
        found = []
        if kept < keep_min:
            found.append({"code": f"{kind}_cut", "detail": f"{kind} {1 - kept:.0%} outside the frame"})
        if under_ui > ui_max:
            found.append({"code": f"{kind}_in_ui_zone", "detail": f"{kind} {under_ui:.0%} under platform UI"})
        return mapped, round(kept, 3), round(under_ui, 3), found

    for face in elements["faces"]:
        mapped, kept, under_ui, found = check("face", face, 0.9, 0.15)
        measures["faces"].append({"box": rounded(mapped), "kept": kept, "under_ui": under_ui})
        issues += found
    if elements["logo"]:
        mapped, kept, under_ui, found = check("logo", elements["logo"], 0.98, 0.05)
        measures["logo"] = {"box": rounded(mapped), "kept": kept, "under_ui": under_ui,
                            "redetected": bool(output_observation["logo"] and output_observation["logo"]["found"])}
        issues += found
        if kept >= 0.98 and not measures["logo"]["redetected"]:
            issues.append({"code": "logo_not_redetected", "detail": "logo is in frame but too small or damaged to be recognised"})
    tallest = 0.0
    for box in elements["text"]:
        mapped, kept, under_ui, found = check("text", box, 0.9, 0.15)
        measures["text"].append({"box": rounded(mapped), "kept": kept, "under_ui": under_ui})
        issues += found
        if kept > 0.5:
            tallest = max(tallest, mapped[3])
    if elements["text"] and tallest and tallest < placement["min_text_px"]:
        issues.append({"code": "text_too_small", "detail": f"tallest text line {tallest:.0f}px, needs {placement['min_text_px']}px"})
    if elements["subject"] and not elements["faces"]:
        mapped = tools.map_box(elements["subject"], transform, placement)
        kept = fraction_inside(mapped, frame)
        measures["subject"] = {"box": rounded(mapped), "kept": round(kept, 3)}
        if kept < 0.7:
            issues.append({"code": "subject_cut", "detail": f"main subject {1 - kept:.0%} outside the frame"})

    # issues of one kind are reported once, with the worst detail kept
    unique = {}
    for issue in issues:
        unique.setdefault(issue["code"], issue)
    measures["redetected_faces"] = len(output_observation["faces"]["items"])
    measures["redetected_text_boxes"] = len(output_observation["text"]["boxes"])
    return list(unique.values()), measures


def severity(issues):
    return sum(SEVERITY.get(issue["code"], 1) for issue in issues)


# ---------------------------------------------------------------- the loop
class Trace:
    def __init__(self):
        self.steps = []
        self.started = time.perf_counter()

    def add(self, **step):
        step["step"] = len(self.steps) + 1
        step["t_ms"] = round((time.perf_counter() - self.started) * 1000, 1)
        self.steps.append(step)


def summarize_source(observation):
    return {
        "quality": observation["quality"],
        "text_share": observation["text"]["share"],
        "text_boxes": len(observation["text"]["boxes"]),
        "faces": len(observation["faces"]["items"]),
        "logo_found": bool(observation["logo"] and observation["logo"]["found"]),
        "subject": observation["subject"]["box"],
    }


def run_source_stage(image, logo, planner, trace):
    work = image
    observation = perceive(work, logo)
    first_observation = observation
    fixes, tried = [], []
    escalation = None
    for iteration in range(MAX_SOURCE_STEPS):
        issues = source_issues(observation, logo is not None)
        context = {"stage": "source", "iteration": iteration, "issues": issues, "tried": tried,
                   "metrics": summarize_source(observation)}
        decision = planner.decide(context)
        step = {"stage": "source", "placement": None, "observation": context["metrics"], "issues": issues, "decision": decision}
        if decision["action"] == "accept":
            trace.add(**step, result="accepted")
            break
        if decision["action"] == "escalate":
            escalation = {"reason": decision["reason"], "issues": issues}
            trace.add(**step, result="escalated to human")
            break
        work = tools.enhance_exposure(work) if decision["action"] == "enhance_exposure" else tools.sharpen(work)
        fixes.append(decision["action"])
        tried.append({"action": decision["action"], "args": decision.get("args", {})})
        observation = perceive(work, logo)
        after = source_issues(observation, logo is not None)
        trace.add(**step, result={"tool": decision["action"], "new_issues": [i["code"] for i in after],
                                  "new_quality": observation["quality"]})
    else:
        escalation = {"reason": "step limit reached", "issues": source_issues(observation, logo is not None)}
    return {"image": work, "observation": observation, "first_observation": first_observation,
            "fixes": fixes, "escalation": escalation}


def light_observation(image):
    """Faces only, for the many per-shot video candidates (text re-detection is the slow part)."""
    from .perception import detect_faces

    kind, faces = detect_faces(image)
    return {"text": {"boxes": [], "share": 0.0}, "faces": {"detector": kind, "items": faces},
            "subject": {"box": None}, "logo": None}


def observe_output(output, logo, elements, transform, placement, full=True):
    """Re-run perception on a rendered candidate.

    The logo check looks where the logo should have landed (2x its box) and
    enlarges small regions first: the question is "is it still recognisable
    there?", not "is there a logo somewhere".
    """
    observation = perceive(output) if full else light_observation(output)
    observation["logo"] = None
    if logo is None or not elements["logo"]:
        return observation
    height, width = output.shape[:2]
    region = expand(tools.map_box(elements["logo"], transform, placement), 2.0, width, height)
    x, y, w, h = (int(round(v)) for v in region)
    patch = output[max(0, y):y + h, max(0, x):x + w]
    if patch.size == 0:
        observation["logo"] = {"found": False, "box": None, "method": None, "score": 0.0}
        return observation
    zoom = min(3.0, max(1.0, 400 / max(patch.shape[:2])))  # capped: a speck blown up is not "legible"
    if zoom > 1:
        patch = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    found = find_logo(patch, logo)
    if found["found"]:
        bx, by, bw, bh = found["box"]
        found["box"] = [max(0, x) + bx / zoom, max(0, y) + by / zoom, bw / zoom, bh / zoom]
        observation["text"]["boxes"] = [b for b in observation["text"]["boxes"] if fraction_inside(b, found["box"]) < 0.6]
    observation["logo"] = found
    return observation


def run_placement(name, work, observation, logo, planner, trace, elements=None, full_recheck=True):
    placement = PLACEMENTS[name]
    elements = elements or elements_of(observation)
    width, height = elements["size"]
    aspect_match = abs((width / height) / (placement["size"][0] / placement["size"][1]) - 1) < ASPECT_TOLERANCE
    priority_differs = tools.keep_box(elements, "priority") != tools.keep_box(elements, "all")
    candidates, tried = [], []
    current = None
    outcome = None
    for iteration in range(MAX_PLACEMENT_STEPS):
        issues = current["issues"] if current else ([] if aspect_match else [{"code": "aspect_mismatch", "detail": f"source {width}x{height}"}])
        context = {"stage": "placement", "placement": name, "target_size": placement["size"],
                   "unsafe_zones_top_bottom_left_right": placement["unsafe"], "iteration": iteration,
                   "aspect_match": aspect_match, "priority_differs": priority_differs,
                   "has_candidate": current is not None,
                   "candidate_kind": current["transform"]["kind"] if current else None,
                   "issues": issues, "tried": tried}
        decision = planner.decide(context)
        step = {"stage": "placement", "placement": name, "observation": current["measures"] if current else None,
                "issues": issues, "decision": decision}
        if decision["action"] == "accept":
            trace.add(**step, result="accepted")
            outcome = "accepted"
            break
        if decision["action"] == "escalate":
            trace.add(**step, result="escalated to human")
            outcome = "escalated"
            break
        transform, fits = _call_reframe_tool(decision, elements, placement)
        output = tools.render(work, transform, placement)
        output_observation = observe_output(output, logo, elements, transform, placement, full_recheck)
        new_issues, measures = placement_issues(elements, transform, output_observation, placement)
        current = {"transform": transform, "image": output, "observation": output_observation,
                   "issues": new_issues, "measures": measures, "action": decision["action"]}
        candidates.append(current)
        tried.append({"action": decision["action"], "args": decision.get("args", {})})
        trace.add(**step, result={"tool": decision["action"], "args": decision.get("args", {}), "transform": transform,
                                  "keep_box_fits_safe_zone": fits, "new_issues": [i["code"] for i in new_issues]})
    else:
        outcome = "escalated"
        trace.add(stage="placement", placement=name, observation=None, issues=current["issues"] if current else [],
                  decision={"action": "escalate", "reason": "step limit reached"}, result="escalated to human")

    best = current if outcome == "accepted" else min(candidates, key=lambda c: severity(c["issues"]), default=None)
    if best is None:  # escalated before any render: still hand the human a proposal
        transform, _ = tools.smart_crop(elements, placement, "all")
        output = tools.render(work, transform, placement)
        output_observation = observe_output(output, logo, elements, transform, placement, full_recheck)
        issues, measures = placement_issues(elements, transform, output_observation, placement)
        best = {"transform": transform, "image": output, "observation": output_observation,
                "issues": issues, "measures": measures, "action": "smart_crop"}
    return {"placement": name, "outcome": outcome, "steps": len(tried), "best": best}


def _call_reframe_tool(decision, elements, placement):
    if decision["action"] == "resize":
        return tools.identity_resize(elements, placement), True
    if decision["action"] == "fit_into_safe_zone":
        return tools.fit_into_safe_zone(elements, placement), True
    return tools.smart_crop(elements, placement, decision.get("args", {}).get("strategy", "all"))


def final_status(source, placement_result):
    reasons = []
    if source["escalation"]:
        reasons.append("source: " + source["escalation"]["reason"])
    if placement_result["outcome"] != "accepted":
        reasons.append("placement: " + (", ".join(i["code"] for i in placement_result["best"]["issues"]) or "unresolved"))
    if reasons:
        return "needs_human", reasons
    changed = source["fixes"] or placement_result["best"]["action"] != "resize"
    return ("auto_fixed" if changed else "pass"), []


def run_image(image, placements, logo=None, planner=None):
    from .planner import RulePlanner

    planner = planner or RulePlanner()
    trace = Trace()
    source = run_source_stage(image, logo, planner, trace)
    results = []
    for name in placements:
        result = run_placement(name, source["image"], source["observation"], logo, planner, trace)
        result["status"], result["reasons"] = final_status(source, result)
        results.append(result)
    return {"source": source, "placements": results, "trace": trace.steps,
            "planner": getattr(planner, "name", "rules"), "runtime_ms": round((time.perf_counter() - trace.started) * 1000, 1)}
