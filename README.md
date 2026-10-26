# AdLint Agent

**An OpenCV 5 agent that checks ad creatives before launch.** Give it a creative (image or short video), an optional brand
logo and the placements you will buy (Stories 9:16, Reels 9:16, Feed 1:1 / 4:5, YouTube 16:9, Display 300x250).
For each placement it looks at the creative, decides what to do, does it, **looks again at its own output**, and
either ships a ready asset or stops and asks a human, with an annotated preview and the reason.

Entry for the OpenCV AI Competition 2026 (powered by AWS). Author: Osama Romoh (solo).

- **Live demo (AWS Lambda, eu-north-1):** https://hkuqxujx6iftfyt6gcq2q64ewm0kndgt.lambda-url.eu-north-1.on.aws/
- Technical report: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) · evaluation: [eval/results_test/results.md](eval/results_test/results.md)
- Diagrams: [docs/architecture.png](docs/architecture.png), [docs/agent_workflow.png](docs/agent_workflow.png) (sources: `docs/*.dot`)
- AWS resources: [AWS_RESOURCES.md](AWS_RESOURCES.md)

## What it checks and does

| perception (OpenCV 5) | used for |
|---|---|
| DB text detector (`cv2.dnn.TextDetectionModel_DB`, PP-OCRv3 ONNX), MSER fallback | text share (text-heavy), headline size per placement, text under platform UI |
| YuNet face detector (`cv2.FaceDetectorYN`), Haar fallback | faces cut by a crop or under platform UI (detection only, no recognition) |
| spectral-residual saliency (`cv2.saliency`) | main-subject box when there is no face |
| ORB + BFMatcher + RANSAC homography (`cv2.findHomography`), template-matching fallback | is the supplied logo there; after reframing, is it still recognisable where it landed |
| contrast-normalised Laplacian sharpness, luma statistics | blurry / severely blurry / dark / flat / blown-out |
| HSV-histogram Bhattacharyya distance (video) | scene cuts, dark/blurry frames, first-3-seconds checks |

| action tool | when the planner picks it |
|---|---|
| `enhance_exposure` (gamma + CLAHE on L) | dark or flat master, then re-checked |
| `sharpen` (unsharp mask) | mildly soft master only, then re-checked |
| `resize` | aspect already matches |
| `smart_crop(all / priority)` | aspect differs; crop window solved so faces, logo, text, subject land in the placement's safe zone |
| `fit_into_safe_zone` | no crop can keep it all: shrink into the safe zone over a blurred backdrop |
| `escalate` | nothing left that fixes it, or the master itself is not shippable |

The loop is bounded (4 steps for the master, 6 per placement) and every step is logged to `trace.json`:
observation → decision (with reason and which planner made it) → tool call → new observation.

Two planners share one interface: a deterministic **rule policy** (default, always works) and an optional **LLM planner**
(OpenRouter; default model `deepseek/deepseek-v4.1-flash`, set `ADLINT_LLM_MODEL`) that receives the structured
observations and returns the next tool call as JSON. Invalid, repeated or failed LLM answers fall back to the rules
and the trace says so.

## Run it locally

Python 3.12, Linux or macOS. All dependencies are pinned in `requirements.txt` / `requirements-dev.txt`
(OpenCV is `opencv-contrib-python-headless==5.0.0.93`, contrib is needed for `cv2.saliency`).
The two ONNX models (YuNet 2023mar, PP-OCRv3 DB text detector, from `opencv/opencv_zoo`) are in `models/`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# CLI: exit code 0 = ready, 2 = needs human
.venv/bin/python -m adlint.cli demo/assets/1_wide_hero.jpg --logo demo/assets/logo_nova.png \
    --placements stories_9x16,feed_1x1,youtube_16x9,display_300x250 --out out/hero
open out/hero/report.html

# showcase run (4 images + 1 video) -> demo_out/
./demo.sh                       # PLANNER=llm OPENROUTER_API_KEY=... ./demo.sh for the LLM planner

# web app
.venv/bin/uvicorn webapp.app:app --port 8080     # http://localhost:8080

# tests
.venv/bin/python -m pytest -q tests
```

Docker (same web app): `docker build -t adlint . && docker run -p 8080:8080 adlint`. Not built yet: our build host blocks container builds, so the Dockerfile is untested.

## Evaluation

```bash
.venv/bin/python eval/make_dataset.py --seed 2026 --out eval/dataset           # dev set (thresholds tuned here)
.venv/bin/python eval/make_dataset.py --seed 2027 --out eval/dataset_holdout   # first held-out run, then error analysis
.venv/bin/python eval/make_dataset.py --seed 2028 --out eval/dataset_test      # final test set (reported)
.venv/bin/python eval/run_eval.py --dataset eval/dataset_test --out eval/results_test
OPENROUTER_API_KEY=... .venv/bin/python eval/run_eval.py --dataset eval/dataset_test --out eval/results_test_llm --planner llm --limit 20
.venv/bin/python eval/video_eval.py --seed 8 --out eval/results_video        # scene cuts / bad frames on synthetic clips
```

60 synthetic creatives per set with known defects (blur, dark, text-heavy, logo near the edge / missing / a different
brand's logo, off-centre subject, four aspect ratios), built from scikit-image's public-domain / CC0 sample photos.
Results, per-item CSVs and failure-case images are in `eval/results_*`. Headline numbers are in TECHNICAL_REPORT.md.

## Deploy to AWS

Lambda (arm64 / Graviton, zip package, no Docker needed) + public Function URL + private S3 bucket for runs.

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=eu-north-1
aws/deploy.sh        # idempotent: creates or updates; prints the URL
aws/teardown.sh      # removes everything it created
```

Environment variables: `ADLINT_S3_BUCKET` (S3 mode; unset = local disk), `ADLINT_WORK_DIR`, `ADLINT_MAX_UPLOAD_MB`,
`ADLINT_MODELS_DIR`, `OPENROUTER_API_KEY` (enables the LLM planner; not set in the cloud deployment),
`ADLINT_LLM_MODEL`.

## Layout

```
adlint/        perception.py  tools.py  planner.py  agent.py  video.py  report.py  placements.py  cli.py
webapp/        app.py (FastAPI)  index.html  storage.py (local disk or S3)
lambda_handler.py  Dockerfile  aws/ (build_lambda.sh, deploy.sh, teardown.sh)
eval/          make_dataset.py  run_eval.py  video_eval.py  results_test/  results_test_llm/  results_holdout_v1/  results_video/
demo/          make_assets.py  assets/   demo.sh -> demo_out/
docs/          architecture.(dot|png)  agent_workflow.(dot|png)
tests/
```

## Image credits

Sample photos from scikit-image (`skimage.data`): astronaut (NASA, public domain), camera (CC0, Lav Varshney),
chelsea and coffee (CC0, Stefan van der Walt / Rachel Michetti), rocket (SpaceX, public domain). Backgrounds, copy,
logos ("NOVA", "ZENTA" are made up) and the bottle are drawn in code.
