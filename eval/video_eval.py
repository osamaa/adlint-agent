"""Scene-cut and bad-frame detection on synthetic clips with known cut frames.

Clips 0-5: hard cuts between 3-5 shots (slow zoom inside each shot, one dark and
one blurred shot per clip). Clip 6: the same, but every transition is a 12-frame
cross-fade, a known hard case for histogram-difference cut detection.

    python eval/video_eval.py --out eval/results_video
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adlint.video import _blurry, blur_threshold, scan  # noqa: E402
from adlint.perception import THRESHOLDS  # noqa: E402
from eval.make_dataset import BASES, base_photo  # noqa: E402

FPS, SIZE, TOLERANCE = 24, (960, 540), 2


def shot_frames(photo, frames, rng, dark=False, blurred=False):
    background = np.full((SIZE[1], SIZE[0], 3), [rng.randint(30, 220) for _ in range(3)], np.uint8)
    out = []
    for i in range(frames):
        canvas = background.copy()
        width = int(SIZE[0] * (0.55 + 0.002 * i))
        height = int(photo.shape[0] * width / photo.shape[1])
        height = min(height, SIZE[1] - 20)
        image = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
        if blurred:
            image = cv2.GaussianBlur(image, (0, 0), 4)
        x, y = (SIZE[0] - width) // 2, (SIZE[1] - height) // 2
        canvas[y:y + height, x:x + width] = image
        out.append((canvas * 0.25).astype(np.uint8) if dark else canvas)
    return out


def make_clip(path, rng, fade=False):
    names = rng.sample(sorted(BASES), rng.randint(3, 5))
    dark_shot, blur_shot = rng.sample(range(len(names)), 2)
    shots = [shot_frames(base_photo(n), rng.randint(30, 60), rng, i == dark_shot, i == blur_shot) for i, n in enumerate(names)]
    frames, cuts, dark, blurry = [], [], 0, 0
    for i, shot in enumerate(shots):
        if i and fade:
            previous = frames[-12:]
            for k in range(12):  # blend the last 12 frames of the previous shot into the next one
                alpha = (k + 1) / 13
                frames[-12 + k] = cv2.addWeighted(previous[k], 1 - alpha, shot[0], alpha, 0)
        if i:
            cuts.append(len(frames))
        frames += shot
        dark += len(shot) if i == dark_shot else 0
        blurry += len(shot) if i == blur_shot else 0
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()
    return {"cuts": cuts, "frames": len(frames), "dark_share": dark / len(frames), "blurry_share": blurry / len(frames)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/results_video")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for index in range(7):
        fade = index == 6
        path = os.path.join(args.out, "clips", f"clip{index}{'_fade' if fade else ''}.mp4")
        truth = make_clip(path, rng, fade)
        found = scan(path)
        predicted = found["cuts"][1:]
        matched = sum(any(abs(p - c) <= TOLERANCE for p in predicted) for c in truth["cuts"])
        false_cuts = sum(not any(abs(p - c) <= TOLERANCE for c in truth["cuts"]) for p in predicted)
        frames = found["frames"]
        rows.append({
            "clip": os.path.basename(path), "fade_transitions": fade, "true_cuts": truth["cuts"], "found_cuts": predicted,
            "cuts_matched": matched, "cuts_missed": len(truth["cuts"]) - matched, "false_cuts": false_cuts,
            "dark_share_true": round(truth["dark_share"], 3),
            "dark_share_found": round(float(np.mean([f["brightness"] < THRESHOLDS["brightness_dark"] for f in frames])), 3),
            "blurry_share_true": round(truth["blurry_share"], 3),
            "blurry_share_found": round(float(np.mean([_blurry(f, blur_threshold(frames)) for f in frames])), 3),
        })
        print(rows[-1])
    hard = [r for r in rows if not r["fade_transitions"]]
    summary = {
        "hard_cut_clips": len(hard),
        "cut_recall": sum(r["cuts_matched"] for r in hard) / sum(len(r["true_cuts"]) for r in hard),
        "cut_precision": sum(r["cuts_matched"] for r in hard) / max(1, sum(r["cuts_matched"] + r["false_cuts"] for r in hard)),
        "fade_clip": rows[-1],
        "tolerance_frames": TOLERANCE, "rows": rows,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"# Video checks on synthetic clips (seed {args.seed})", "",
             f"Hard cuts ({len(hard)} clips): recall {summary['cut_recall']:.2f}, precision {summary['cut_precision']:.2f} (±{TOLERANCE} frames)", "",
             "| clip | true cuts | found cuts | missed | false | dark share true/found | blurry share true/found |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['clip']} | {r['true_cuts']} | {r['found_cuts']} | {r['cuts_missed']} | {r['false_cuts']} | "
              f"{r['dark_share_true']:.2f} / {r['dark_share_found']:.2f} | {r['blurry_share_true']:.2f} / {r['blurry_share_found']:.2f} |" for r in rows]
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
