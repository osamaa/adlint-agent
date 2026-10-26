"""Turn an agent run into files: final assets, annotated previews, report.json, report.html, trace.json."""

import html
import json
import os

import cv2
import numpy as np

from .placements import PLACEMENTS, safe_rect, unsafe_rects

COLORS = {"text": (0, 215, 255), "face": (80, 220, 80), "logo": (255, 140, 0), "subject": (255, 0, 255)}
STATUS_COLORS = {"pass": (80, 180, 80), "auto_fixed": (200, 140, 40), "needs_human": (40, 40, 220)}


def annotate(image, observation, placement=None, title=None, status=None):
    """Draw what perception found; for a placement also shade the UI-covered zones."""
    zoom = max(1.0, 600 / max(image.shape[:2]))  # small placements are drawn enlarged
    canvas = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC) if zoom > 1 else image.copy()
    scale = max(canvas.shape[:2]) / 1000
    thickness = max(1, round(2.5 * scale))
    if placement:
        shade = canvas.copy()
        for x, y, w, h in unsafe_rects(placement):
            x, y, w, h = (int(v * zoom) for v in (x, y, w, h))
            cv2.rectangle(shade, (x, y), (x + w, y + h), (0, 0, 255), -1)
        canvas = cv2.addWeighted(shade, 0.28, canvas, 0.72, 0)
        x, y, w, h = (int(v * zoom) for v in safe_rect(placement))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), thickness)

    def draw(box, color, label):
        x, y, w, h = (int(v * zoom) for v in box)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(canvas, label, (x + 2, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, color, thickness)

    for box in observation["text"]["boxes"]:
        draw(box, COLORS["text"], "text")
    for face in observation["faces"]["items"]:
        draw(face["box"], COLORS["face"], f"face {face['score']:.2f}")
    if observation["subject"]["box"]:
        draw(observation["subject"]["box"], COLORS["subject"], "subject")
    if observation["logo"] and observation["logo"]["found"]:
        draw(observation["logo"]["box"], COLORS["logo"], f"logo ({observation['logo']['method']})")

    if title:
        bar_scale = canvas.shape[1] / 1000
        bar_h = round(50 * bar_scale)
        bar = np.full((bar_h, canvas.shape[1], 3), STATUS_COLORS.get(status, (60, 60, 60)), np.uint8)
        cv2.putText(bar, title, (8, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * bar_scale,
                    (255, 255, 255), max(1, round(2 * bar_scale)))
        canvas = np.vstack([bar, canvas])
    return canvas


def _clean(value):
    """Make numpy values JSON-safe."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_outputs(run, out_dir, source_name="creative"):
    os.makedirs(out_dir, exist_ok=True)
    source = run["source"]
    cv2.imwrite(os.path.join(out_dir, "source_annotated.jpg"),
                annotate(source["image"], source["observation"], title="source: what the agent sees" + (" (after " + ", ".join(source["fixes"]) + ")" if source["fixes"] else "")))
    placements = []
    for result in run["placements"]:
        name, best = result["placement"], result["best"]
        asset = f"{name}.jpg" if result["status"] != "needs_human" else f"{name}_PROPOSED.jpg"
        cv2.imwrite(os.path.join(out_dir, asset), best["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        title = f"{name}: {result['status'].upper()} ({best['action']})"
        preview = annotate(best["image"], best["observation"], PLACEMENTS[name], title, result["status"])
        cv2.imwrite(os.path.join(out_dir, f"{name}_preview.jpg"), preview)
        placements.append({
            "placement": name, "label": PLACEMENTS[name]["label"], "status": result["status"],
            "reasons": result["reasons"], "final_tool": best["action"], "transform": best["transform"],
            "issues": best["issues"], "checks": best["measures"], "agent_steps": result["steps"],
            "asset": asset, "preview": f"{name}_preview.jpg",
        })
    report = _clean({
        "creative": source_name,
        "planner": run["planner"],
        "runtime_ms": run["runtime_ms"],
        "source": {
            "size": source["observation"]["size"],
            "quality_before": source["first_observation"]["quality"],
            "quality_after": source["observation"]["quality"],
            "fixes_applied": source["fixes"],
            "escalation": source["escalation"],
            "text_share": source["observation"]["text"]["share"],
            "faces": len(source["observation"]["faces"]["items"]),
            "logo": source["observation"]["logo"],
            "detectors": {"text": source["observation"]["text"]["detector"],
                          "faces": source["observation"]["faces"]["detector"]},
        },
        "placements": placements,
        "needs_human": any(p["status"] == "needs_human" for p in placements),
    })
    if source["fixes"]:
        cv2.imwrite(os.path.join(out_dir, "source_fixed.jpg"), source["image"])
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(_clean(run["trace"]), f, indent=2)
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(render_html(report, _clean(run["trace"])))
    return report


def render_html(report, trace):
    e = html.escape
    rows = []
    for p in report["placements"]:
        issues = ", ".join(i["code"] for i in p["issues"]) or "none"
        reasons = "<br>".join(e(r) for r in p["reasons"]) or "&mdash;"
        rows.append(f"""<div class="card {p['status']}"><h3>{e(p['label'])} <span>{p['status'].replace('_', ' ')}</span></h3>
<img src="{p['preview']}"><p><b>final tool:</b> {e(p['final_tool'])} &middot; <b>agent steps:</b> {p['agent_steps']}<br>
<b>remaining issues:</b> {e(issues)}<br><b>why human:</b> {reasons}</p><a href="{p['asset']}">download asset</a></div>""")
    steps = []
    for s in trace:
        issues = ", ".join(i["code"] for i in s["issues"]) or "none"
        result = s["result"] if isinstance(s["result"], str) else "new issues: " + (", ".join(s["result"]["new_issues"]) or "none")
        steps.append(f"<tr><td>{s['step']}</td><td>{e(s['placement'] or 'source')}</td><td>{e(issues)}</td>"
                     f"<td><b>{e(s['decision']['action'])}</b> {e(json.dumps(s['decision'].get('args', {})))}<br><small>{e(s['decision'].get('reason', ''))}</small></td>"
                     f"<td>{e(result)}</td></tr>")
    src = report["source"]
    esc = e(src["escalation"]["reason"]) if src["escalation"] else "none"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>AdLint report: {e(report['creative'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}} .grid{{display:flex;flex-wrap:wrap;gap:16px}}
.card{{border:2px solid #ccc;border-radius:8px;padding:12px;width:360px}} .card img{{width:100%}} .card h3 span{{float:right;font-size:13px;padding:2px 8px;border-radius:10px;color:#fff;background:#777}}
.pass h3 span{{background:#3a3}} .auto_fixed h3 span{{background:#c80}} .needs_human h3 span{{background:#c22}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ddd;padding:4px 8px;vertical-align:top;text-align:left}}</style></head><body>
<h1>AdLint report: {e(report['creative'])}</h1>
<p><b>Needs human approval:</b> {'YES' if report['needs_human'] else 'no'} &middot; planner: {e(report['planner'])} &middot; runtime {report['runtime_ms']/1000:.1f}s</p>
<p><b>Source:</b> {src['size'][0]}x{src['size'][1]} &middot; fixes applied: {e(', '.join(src['fixes_applied']) or 'none')} &middot; source escalation: {esc}<br>
sharpness {src['quality_before']['sharpness']} &rarr; {src['quality_after']['sharpness']} &middot; brightness {src['quality_before']['brightness']} &rarr; {src['quality_after']['brightness']} &middot; text covers {src['text_share']:.0%} &middot; faces {src['faces']} &middot; logo found: {bool(src['logo'] and src['logo']['found'])}</p>
<img src="source_annotated.jpg" style="max-width:480px">
<h2>Placements</h2><div class="grid">{''.join(rows)}</div>
<h2>Agent trace (observation &rarr; decision &rarr; tool &rarr; new observation)</h2>
<table><tr><th>#</th><th>stage</th><th>observed issues</th><th>decision</th><th>result</th></tr>{''.join(steps)}</table>
</body></html>"""
