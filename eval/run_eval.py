"""Evaluate AdLint on the labelled synthetic set.

Measures (1) detection precision/recall per check, (2) reframing success per
placement for a naive centre crop vs the agent loop, judged on ground-truth
boxes (not on what the agent itself perceived), (3) steps and runtime.

    python eval/run_eval.py --dataset eval/dataset --out eval/results [--planner rules|llm] [--limit N]
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adlint import tools  # noqa: E402
from adlint.agent import run_image  # noqa: E402
from adlint.geometry import fraction_inside, intersection, iou  # noqa: E402
from adlint.perception import THRESHOLDS  # noqa: E402
from adlint.placements import PLACEMENTS, safe_rect  # noqa: E402
from adlint.planner import make_planner  # noqa: E402
from adlint.report import annotate  # noqa: E402

EVAL_PLACEMENTS = ["stories_9x16", "reels_9x16", "feed_1x1", "youtube_16x9", "display_300x250"]
CHECKS = ["blurry", "severely_blurry", "dark", "text_heavy", "logo_present", "face_present"]


def predicted_flags(observation):
    flags = observation["flags"]
    return {
        "blurry": flags["blurry"] or flags["severely_blurry"],
        "severely_blurry": flags["severely_blurry"],
        "dark": flags["dark"],
        "text_heavy": observation["text"]["share"] > THRESHOLDS["text_heavy_share"],
        "logo_present": bool(observation["logo"] and observation["logo"]["found"]),
        "face_present": len(observation["faces"]["items"]) > 0,
    }


def centre_crop(size, placement):
    width, height = size
    aspect = placement["size"][0] / placement["size"][1]
    crop_w = min(width, height * aspect)
    crop_h = crop_w / aspect
    return {"kind": "crop", "rect": [(width - crop_w) / 2, (height - crop_h) / 2, crop_w, crop_h]}


def judge(truth, transform, placement):
    """Ground-truth verdict for one output framing."""
    width, height = placement["size"]
    frame, safe = [0, 0, width, height], safe_rect(placement)

    def kept_and_ui(box):
        mapped = tools.map_box(box, transform, placement)
        visible = intersection(mapped, frame)
        return fraction_inside(mapped, frame), (1 - fraction_inside(visible, safe)) if visible else 0.0, mapped

    reasons = []
    for face in truth["faces"]:
        kept, ui, _ = kept_and_ui(face)
        if kept < 0.9 or ui > 0.15:
            reasons.append("face")
    if truth["logo"]:
        kept, ui, _ = kept_and_ui(truth["logo"])
        if kept < 0.98 or ui > 0.05:
            reasons.append("logo")
    kept, ui, mapped = kept_and_ui(truth["headline"])
    if kept < 0.9 or ui > 0.15:
        reasons.append("headline")
    legible = mapped[3] >= placement["min_text_px"]
    kept, _, _ = kept_and_ui(truth["subject"])
    if kept < 0.7:
        reasons.append("subject")
    return {"ok": not reasons, "failed": sorted(set(reasons)), "headline_legible": legible}


def prf(rows, check):
    tp = sum(1 for r in rows if r[f"gt_{check}"] and r[f"pred_{check}"])
    fp = sum(1 for r in rows if not r[f"gt_{check}"] and r[f"pred_{check}"])
    fn = sum(1 for r in rows if r[f"gt_{check}"] and not r[f"pred_{check}"])
    tn = len(rows) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"check": check, "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall}


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--planner", default="rules", choices=["rules", "llm"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "failures"), exist_ok=True)

    data = json.load(open(os.path.join(args.dataset, "labels.json")))
    logo = cv2.imread(os.path.join(args.dataset, data["logo"]))
    items = data["items"][: args.limit] if args.limit else data["items"]
    planner = make_planner(args.planner)

    item_rows, placement_rows, failures = [], [], []
    for item in items:
        image = cv2.imread(os.path.join(args.dataset, item["file"]))
        started = time.perf_counter()
        run = run_image(image, EVAL_PLACEMENTS, logo, planner)
        seconds = time.perf_counter() - started
        observation = run["source"]["first_observation"]
        predicted = predicted_flags(observation)
        labels = item["labels"]
        expected_human = labels["severely_blurry"] or labels["text_heavy"] or not labels["logo_present"]
        row = {"file": item["file"], "seconds": round(seconds, 2), "expected_source_escalation": expected_human,
               "source_escalated": run["source"]["escalation"] is not None,
               "source_fixes": "|".join(run["source"]["fixes"]), **{f"gt_{k}": item["labels"][k] for k in CHECKS},
               **{f"pred_{k}": predicted[k] for k in CHECKS}}
        truth = item["truth"]
        if truth["faces"] and observation["faces"]["items"]:
            row["face_iou"] = round(max(iou(truth["faces"][0], f["box"]) for f in observation["faces"]["items"]), 3)
        if truth["logo"] and predicted["logo_present"]:
            row["logo_iou"] = round(iou(truth["logo"], observation["logo"]["box"]), 3)
        item_rows.append(row)
        wrong = [k for k in CHECKS if item["labels"][k] != predicted[k]]
        if wrong:
            failures.append({"file": item["file"], "kind": "detection", "wrong": wrong,
                             "quality": observation["quality"], "text_share": observation["text"]["share"]})
            cv2.imwrite(os.path.join(args.out, "failures", f"det_{item['file'].replace('.png', '.jpg')}"),
                        annotate(image, observation, title="detection miss: " + ",".join(wrong), status="needs_human"))

        for result in run["placements"]:
            placement = PLACEMENTS[result["placement"]]
            baseline = judge(truth, centre_crop(observation["size"], placement), placement)
            agent = judge(truth, result["best"]["transform"], placement)
            accepted = result["status"] != "needs_human"
            placement_rows.append({
                "file": item["file"], "placement": result["placement"], "status": result["status"],
                "final_tool": result["best"]["action"], "steps": result["steps"],
                "baseline_ok": baseline["ok"], "baseline_failed": "|".join(baseline["failed"]),
                "agent_ok": agent["ok"], "agent_failed": "|".join(agent["failed"]),
                "agent_headline_legible": agent["headline_legible"],
                "false_accept": accepted and not agent["ok"],
                "agent_issues": "|".join(i["code"] for i in result["best"]["issues"]),
                "reasons": " / ".join(result["reasons"]),
            })
            if accepted and not agent["ok"]:
                failures.append({"file": item["file"], "kind": "false_accept", "placement": result["placement"],
                                 "gt_failed": agent["failed"], "agent_issues": [i["code"] for i in result["best"]["issues"]]})
                cv2.imwrite(os.path.join(args.out, "failures", f"fa_{result['placement']}_{item['file'].replace('.png', '.jpg')}"),
                            annotate(result["best"]["image"], result["best"]["observation"], placement,
                                     "false accept: GT " + ",".join(agent["failed"]), "needs_human"))
        print(f"{item['file']}: {seconds:.1f}s " + " ".join(f"{r['placement']}={r['status']}" for r in run["placements"]), flush=True)

    detection = [prf(item_rows, check) for check in CHECKS]
    per_placement = []
    for name in EVAL_PLACEMENTS:
        rows = [r for r in placement_rows if r["placement"] == name]
        accepted = [r for r in rows if r["status"] != "needs_human"]
        per_placement.append({
            "placement": name, "n": len(rows),
            "centre_crop_ok": sum(r["baseline_ok"] for r in rows) / len(rows),
            "agent_ok": sum(r["agent_ok"] for r in rows) / len(rows),
            "agent_auto_accepted": len(accepted) / len(rows),
            "agent_ok_when_accepted": (sum(r["agent_ok"] for r in accepted) / len(accepted)) if accepted else None,
            "false_accepts": sum(r["false_accept"] for r in rows),
            "headline_legible": sum(r["agent_headline_legible"] for r in rows) / len(rows),
            "mean_steps": statistics.mean(r["steps"] for r in rows),
            "max_steps": max(r["steps"] for r in rows),
        })
    seconds = [r["seconds"] for r in item_rows]
    escalation = prf([{"gt_x": r["expected_source_escalation"], "pred_x": r["source_escalated"]} for r in item_rows], "x")
    escalation["check"] = "source escalated to human"
    face_ious = [r["face_iou"] for r in item_rows if "face_iou" in r]
    logo_ious = [r["logo_iou"] for r in item_rows if "logo_iou" in r]
    summary = {
        "planner": args.planner, "creatives": len(items), "placements_per_creative": len(EVAL_PLACEMENTS),
        "thresholds": THRESHOLDS, "detection": detection, "source_escalation": escalation, "reframing": per_placement,
        "overall": {
            "centre_crop_ok": sum(r["baseline_ok"] for r in placement_rows) / len(placement_rows),
            "agent_ok": sum(r["agent_ok"] for r in placement_rows) / len(placement_rows),
            "false_accepts": sum(r["false_accept"] for r in placement_rows),
            "outputs": len(placement_rows),
            "escalated": sum(r["status"] == "needs_human" for r in placement_rows),
        },
        "runtime_seconds": {"mean": statistics.mean(seconds), "median": statistics.median(seconds), "max": max(seconds)},
        "face_box_iou_mean": statistics.mean(face_ious) if face_ious else None,
        "logo_box_iou_mean": statistics.mean(logo_ious) if logo_ious else None,
        "failures": failures,
    }

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for name, rows in (("per_creative.csv", item_rows), ("per_placement.csv", placement_rows)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    lines = [f"# AdLint evaluation ({args.planner} planner, {len(items)} creatives x {len(EVAL_PLACEMENTS)} placements)", "",
             "## Detection (source creative)", "", "| check | positives | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {d['check']} | {d['positives']} | {d['tp']} | {d['fp']} | {d['fn']} | {fmt(d['precision'])} | {fmt(d['recall'])} |" for d in detection]
    lines += [f"| *{escalation['check']}* (expected = severe blur, text-heavy or logo missing) | {escalation['positives']} | {escalation['tp']} | "
              f"{escalation['fp']} | {escalation['fn']} | {fmt(escalation['precision'])} | {fmt(escalation['recall'])} |"]
    lines += ["", f"Face box IoU vs ground truth (detected faces): {fmt(summary['face_box_iou_mean'])} · logo box IoU: {fmt(summary['logo_box_iou_mean'])}", "",
              "## Reframing: is everything important kept and clear of platform UI? (judged on ground-truth boxes)", "",
              "| placement | centre crop ok | agent ok | agent auto-accepted | ok when accepted | false accepts | headline legible | mean steps | max steps |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {p['placement']} | {p['centre_crop_ok']:.0%} | {p['agent_ok']:.0%} | {p['agent_auto_accepted']:.0%} | "
              f"{'n/a' if p['agent_ok_when_accepted'] is None else format(p['agent_ok_when_accepted'], '.0%')} | {p['false_accepts']} | "
              f"{p['headline_legible']:.0%} | {p['mean_steps']:.2f} | {p['max_steps']} |" for p in per_placement]
    o = summary["overall"]
    lines += ["", f"**Overall:** centre crop {o['centre_crop_ok']:.0%} ok → agent {o['agent_ok']:.0%} ok; "
              f"{o['escalated']} of {o['outputs']} outputs escalated to a human; {o['false_accepts']} false accepts.",
              f"**Runtime per creative (all {len(EVAL_PLACEMENTS)} placements, CPU):** mean {summary['runtime_seconds']['mean']:.1f}s, "
              f"median {summary['runtime_seconds']['median']:.1f}s, max {summary['runtime_seconds']['max']:.1f}s", "",
              f"## Failure cases ({len(failures)})", ""]
    lines += [f"- `{f['file']}` {f['kind']}: " + (", ".join(f["wrong"]) if f["kind"] == "detection"
              else f"{f['placement']} GT failed {f['gt_failed']}, agent saw {f['agent_issues']}") for f in failures]
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
