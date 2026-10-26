#!/usr/bin/env bash
# Run the showcase creatives end to end. Outputs: demo_out/<name>/ (assets, previews, report.html, trace.json)
# Optional: export OPENROUTER_API_KEY=... and PLANNER=llm to run the LLM planner instead of the rule policy.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
PLANNER="${PLANNER:-rules}"
$PY demo/make_assets.py
LOGO=demo/assets/logo_nova.png
PLACEMENTS=stories_9x16,reels_9x16,feed_1x1,youtube_16x9,display_300x250
for creative in demo/assets/1_wide_hero.jpg demo/assets/2_dark_square.jpg demo/assets/3_story_cta_bottom.jpg \
                demo/assets/4_blurry_text_wall.jpg demo/assets/5_promo.mp4; do
  name=$(basename "${creative%.*}")
  echo "== $name"
  $PY -m adlint.cli "$creative" --logo "$LOGO" --placements "$PLACEMENTS" --planner "$PLANNER" --out "demo_out/$name" 2>/dev/null
  # exit code 2 = "needs human" (a valid outcome, not an error)
  [ $? -le 2 ] || { echo "failed on $creative"; exit 1; }
done
echo "open demo_out/*/report.html"
