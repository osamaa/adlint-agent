# AdLint evaluation (rules planner, 60 creatives x 5 placements)

## Detection (source creative)

| check | positives | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| blurry | 13 | 12 | 4 | 1 | 0.75 | 0.92 |
| severely_blurry | 5 | 4 | 2 | 1 | 0.67 | 0.80 |
| dark | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| text_heavy | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| logo_present | 46 | 35 | 2 | 11 | 0.95 | 0.76 |
| face_present | 14 | 12 | 0 | 2 | 1.00 | 0.86 |
| *source escalated to human* (expected = severe blur, text-heavy or logo missing) | 23 | 22 | 14 | 1 | 0.61 | 0.96 |

Face box IoU vs ground truth (detected faces): 0.92 · logo box IoU: 0.93

## Reframing: is everything important kept and clear of platform UI? (judged on ground-truth boxes)

| placement | centre crop ok | agent ok | agent auto-accepted | ok when accepted | false accepts | headline legible | mean steps | max steps |
|---|---|---|---|---|---|---|---|---|
| stories_9x16 | 0% | 100% | 40% | 100% | 0 | 97% | 2.75 | 3 |
| reels_9x16 | 0% | 98% | 37% | 100% | 0 | 95% | 2.75 | 3 |
| feed_1x1 | 28% | 97% | 40% | 96% | 1 | 97% | 2.03 | 3 |
| youtube_16x9 | 10% | 95% | 32% | 100% | 0 | 83% | 2.73 | 3 |
| display_300x250 | 8% | 93% | 25% | 93% | 1 | 65% | 2.47 | 3 |

**Overall:** centre crop 9% ok → agent 97% ok; 196 of 300 outputs escalated to a human; 2 false accepts.
**Runtime per creative (all 5 placements, CPU):** mean 3.2s, median 3.2s, max 4.7s

## Failure cases (23)

- `c008_1x1_rocket_blur_mild_edge.png` detection: logo_present
- `c009_1x1_chelsea_text_heavy_safe.png` detection: logo_present
- `c010_4x5_astronaut_blur_mild_absent.png` detection: severely_blurry
- `c011_4x5_product_none_distractor.png` detection: logo_present
- `c012_16x9_camera_blur_severe_distractor.png` detection: severely_blurry, logo_present, face_present
- `c017_9x16_chelsea_none_safe.png` detection: logo_present
- `c019_1x1_product_text_heavy_safe.png` detection: blurry
- `c022_16x9_rocket_text_heavy_safe.png` detection: logo_present
- `c029_4x5_rocket_none_safe.png` false_accept: feed_1x1 GT failed ['subject'], agent saw []
- `c029_4x5_rocket_none_safe.png` false_accept: display_300x250 GT failed ['subject'], agent saw []
- `c030_9x16_camera_blur_mild_edge.png` detection: blurry
- `c035_9x16_coffee_none_edge.png` detection: logo_present
- `c036_4x5_coffee_none_edge.png` detection: logo_present
- `c037_1x1_coffee_dark_safe.png` detection: logo_present
- `c040_1x1_product_dark_safe.png` detection: blurry
- `c045_4x5_astronaut_blur_mild_absent.png` detection: severely_blurry
- `c046_1x1_chelsea_none_safe.png` detection: logo_present
- `c047_1x1_camera_dark_safe.png` detection: face_present
- `c052_9x16_coffee_none_edge.png` detection: logo_present
- `c054_1x1_chelsea_dark_absent.png` detection: blurry
- `c055_4x5_coffee_dark_edge.png` detection: logo_present
- `c056_9x16_chelsea_blur_severe_safe.png` detection: logo_present
- `c059_1x1_product_none_edge.png` detection: blurry
