"""Video creatives: frame checks, scene cuts, first-3-seconds checks, per-shot reframing.

The agent plans one crop per shot (a shot = frames between two cuts) using
everything perception saw on that shot's keyframes, renders every placement,
then re-checks the rendered keyframes.
"""

import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from .agent import Trace, final_status, run_placement, severity, source_issues
from .geometry import union_box
from .perception import THRESHOLDS, measure_quality, perceive
from .placements import PLACEMENTS
from .report import _clean, annotate, render_html

MAX_SECONDS = 60
CUT_DISTANCE = 0.45       # Bhattacharyya distance between consecutive HSV histograms
MIN_SHOT_SECONDS = 0.4
KEYFRAMES_PER_SECOND = 1.0
DARK_FRAME_BRIGHTNESS = THRESHOLDS["brightness_dark"]


def _histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def scan(path):
    """One pass over the video: per-frame exposure/sharpness, cut detection, keyframes."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("cannot open video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames, cuts, keyframes = [], [0], {}
    previous = None
    index = 0
    keyframe_every = max(1, round(fps / KEYFRAMES_PER_SECOND))
    while index < MAX_SECONDS * fps:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, round(320 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
        histogram = _histogram(small)
        if previous is not None:
            distance = cv2.compareHist(previous, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if distance > CUT_DISTANCE and index - cuts[-1] >= MIN_SHOT_SECONDS * fps:
                cuts.append(index)
                keyframes[index] = frame  # the first frame of every shot is a keyframe
        previous = histogram
        if index % 2 == 0:
            quality = measure_quality(frame)
            frames.append({"index": index, "brightness": quality["brightness"], "sharpness": quality["sharpness"]})
        if index % keyframe_every == 0:
            keyframes[index] = frame
        index += 1
    capture.release()
    if index == 0:
        raise ValueError("video has no frames")
    size = (keyframes[0].shape[1], keyframes[0].shape[0])
    return {"fps": fps, "frame_count": index, "size": size, "frames": frames, "cuts": cuts, "keyframes": keyframes}


def blur_threshold(frames):
    """Per-clip blur threshold. Codecs soften every frame, so a frame counts as blurry
    when it is far below the clip's own sharp frames (40% of the 90th percentile),
    and always when below the absolute 'severe' level. Capped at the image threshold."""
    values = [f["sharpness"] for f in frames if f["sharpness"] is not None]
    reference = float(np.percentile(values, 90)) if values else 0.0
    return min(THRESHOLDS["sharpness_blurry"], max(THRESHOLDS["sharpness_severe"], 0.4 * reference))


def _blurry(frame, threshold):
    return frame["sharpness"] is not None and frame["sharpness"] < threshold


def _shots(scan_result):
    bounds = scan_result["cuts"] + [scan_result["frame_count"]]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _shot_elements(observations, size):
    """Everything that must survive in this shot, across its keyframes."""
    faces, text, logos, subjects = [], [], [], []
    for observation in observations:
        faces += [f["box"] for f in observation["faces"]["items"]]
        text += observation["text"]["boxes"]
        if observation["logo"] and observation["logo"]["found"]:
            logos.append(observation["logo"]["box"])
        if observation["subject"]["box"]:
            subjects.append(observation["subject"]["box"])
    return {"size": list(size), "faces": faces, "text": text, "logo": union_box(logos), "subject": union_box(subjects)}


def _write_video(path_in, path_out_stem, shot_transforms, placement, fps, render):
    """Render one placement. H.264 mp4 when ffmpeg exists, else VP8 webm (OpenCV's own
    encoder): both play in a browser. Returns the file name written."""
    width, height = placement["size"]
    use_ffmpeg = shutil.which("ffmpeg") is not None
    raw = path_out_stem + (".raw.mp4" if use_ffmpeg else ".webm")
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if use_ffmpeg else "VP80"))
    writer = cv2.VideoWriter(raw, fourcc, fps, (width, height))
    capture = cv2.VideoCapture(path_in)
    index, shot = 0, 0
    while True:
        ok, frame = capture.read()
        if not ok or index >= MAX_SECONDS * fps:
            break
        while shot + 1 < len(shot_transforms) and index >= shot_transforms[shot + 1][0]:
            shot += 1
        writer.write(render(frame, shot_transforms[shot][1], placement))
        index += 1
    capture.release()
    writer.release()
    if not use_ffmpeg:
        return os.path.basename(raw)
    final = path_out_stem + ".mp4"
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", final], capture_output=True, timeout=300)
    if result.returncode != 0:
        os.replace(raw, final)
    else:
        os.remove(raw)
    return os.path.basename(final)


def check_video(path, placements, logo, planner, out_dir):
    from .tools import render

    os.makedirs(out_dir, exist_ok=True)
    started = time.perf_counter()
    trace = Trace()
    scanned = scan(path)
    fps, size = scanned["fps"], scanned["size"]
    duration = scanned["frame_count"] / fps
    observations = {i: perceive(frame, logo) for i, frame in sorted(scanned["keyframes"].items())}

    # ---- video-level checks
    frames = scanned["frames"]
    dark_share = float(np.mean([f["brightness"] < DARK_FRAME_BRIGHTNESS for f in frames]))
    threshold = blur_threshold(frames)
    blurry_share = float(np.mean([_blurry(f, threshold) for f in frames]))
    opening = [o for i, o in observations.items() if i < 3 * fps]
    opening_frames = [f for f in frames if f["index"] < 3 * fps]
    notes = []
    if logo is not None and not any(o["logo"] and o["logo"]["found"] for o in opening):
        notes.append("brand logo not seen in the first 3 seconds")
    if opening_frames and np.mean([f["brightness"] < DARK_FRAME_BRIGHTNESS for f in opening_frames]) > 0.3:
        notes.append("first 3 seconds are dark")
    if opening_frames and np.mean([_blurry(f, threshold) for f in opening_frames]) > 0.3:
        notes.append("first 3 seconds are blurry")
    if not any(o["faces"]["items"] or o["text"]["boxes"] for o in opening):
        notes.append("no face or text in the first 3 seconds (weak hook)")
    video_issues = []
    if dark_share > 0.2:
        video_issues.append({"code": "dark_frames", "detail": f"{dark_share:.0%} of frames are dark"})
    if blurry_share > 0.2:
        video_issues.append({"code": "blurry_frames", "detail": f"{blurry_share:.0%} of frames are blurry"})
    if logo is not None and not any(o["logo"] and o["logo"]["found"] for o in observations.values()):
        video_issues.append({"code": "logo_missing", "detail": "supplied logo never found"})
    video_issues += [{"code": "first_3s", "detail": n} for n in notes if "logo" in n or "dark" in n]
    decision = {"action": "escalate" if video_issues else "accept", "args": {},
                "reason": "; ".join(i["detail"] for i in video_issues) or "frames pass exposure, sharpness and opening checks",
                "planner": "rules (video checks)"}
    trace.add(stage="source", placement=None, issues=video_issues, decision=decision,
              observation={"duration_s": round(duration, 2), "scene_cuts": len(scanned["cuts"]) - 1,
                           "dark_frames_share": round(dark_share, 3), "blurry_frames_share": round(blurry_share, 3)},
              result="escalated to human" if video_issues else "accepted")
    source_state = {"fixes": [], "escalation": {"reason": decision["reason"], "issues": video_issues} if video_issues else None}

    # ---- per placement, per shot: the same perceive -> decide -> act loop as images
    shots = _shots(scanned)
    placement_reports = []
    for name in placements:
        placement = PLACEMENTS[name]
        shot_transforms, shot_results = [], []
        for start, end in shots:
            keys = [i for i in observations if start <= i < end] or [start]
            elements = _shot_elements([observations[i] for i in keys if i in observations], size)
            middle = keys[len(keys) // 2]
            result = run_placement(name, scanned["keyframes"][middle], observations[middle], logo, planner, trace, elements,
                                   full_recheck=False)
            shot_transforms.append((start, result["best"]["transform"]))
            shot_results.append(result)
        worst = max(shot_results, key=lambda r: (r["outcome"] != "accepted", severity(r["best"]["issues"])))
        status, reasons = final_status(source_state, worst)
        stem = name if status != "needs_human" else f"{name}_PROPOSED"
        asset = _write_video(path, os.path.join(out_dir, stem), shot_transforms, placement, fps, render)
        strip = [annotate(r["best"]["image"], r["best"]["observation"], placement) for r in shot_results]
        height = 480
        strip = np.hstack([cv2.resize(s, (round(s.shape[1] * height / s.shape[0]), height)) for s in strip])
        cv2.imwrite(os.path.join(out_dir, f"{name}_preview.jpg"),
                    annotate(strip, {"text": {"boxes": []}, "faces": {"items": []}, "subject": {"box": None}, "logo": None},
                             title=f"{name}: {status.upper()} ({len(shots)} shots)", status=status))
        placement_reports.append({
            "placement": name, "label": placement["label"], "status": status, "reasons": reasons,
            "final_tool": ", ".join(r["best"]["action"] for r in shot_results), "transform": [t for _, t in shot_transforms],
            "issues": worst["best"]["issues"], "checks": [r["best"]["measures"] for r in shot_results],
            "agent_steps": sum(r["steps"] for r in shot_results), "asset": asset, "preview": f"{name}_preview.jpg",
        })

    # ---- contact sheet: one tile per keyframe, cuts in red
    tiles = []
    for i, observation in observations.items():
        tile = annotate(scanned["keyframes"][i], observation, title=f"t={i / fps:.1f}s" + (" CUT" if i in scanned["cuts"][1:] else ""),
                        status="needs_human" if i in scanned["cuts"][1:] else None)
        tiles.append(cv2.resize(tile, (360, round(360 * tile.shape[0] / tile.shape[1]))))
    columns = 6
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    sheet = np.vstack([np.hstack(tiles[r:r + columns]) for r in range(0, len(tiles), columns)])
    cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), sheet)
    cv2.imwrite(os.path.join(out_dir, "source_annotated.jpg"), sheet)

    first = observations[min(observations)]
    report = _clean({
        "creative": os.path.basename(path), "planner": getattr(planner, "name", "rules"),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "video": {"duration_s": round(duration, 2), "fps": round(fps, 2), "size": list(size),
                  "scene_cuts": [round(c / fps, 2) for c in scanned["cuts"][1:]],
                  "dark_frames_share": round(dark_share, 3), "blurry_frames_share": round(blurry_share, 3),
                  "first_3s": {"notes": notes}, "issues": video_issues},
        "source": {"size": list(size), "quality_before": first["quality"], "quality_after": first["quality"],
                   "fixes_applied": [], "escalation": source_state["escalation"], "text_share": first["text"]["share"],
                   "faces": len(first["faces"]["items"]), "logo": first["logo"],
                   "source_issues_first_frame": [i["code"] for i in source_issues(first, logo is not None)]},
        "placements": placement_reports,
        "needs_human": any(p["status"] == "needs_human" for p in placement_reports),
    })
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(_clean(trace.steps), f, indent=2)
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(render_html(report, _clean(trace.steps)))
    return report
