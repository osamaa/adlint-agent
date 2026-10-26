# AdLint evaluation (rules planner, 60 creatives x 5 placements)

## Detection (source creative)

| check | positives | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| blurry | 19 | 18 | 2 | 1 | 0.90 | 0.95 |
| severely_blurry | 6 | 5 | 5 | 1 | 0.50 | 0.83 |
| dark | 15 | 15 | 1 | 0 | 0.94 | 1.00 |
| text_heavy | 5 | 5 | 0 | 0 | 1.00 | 1.00 |
| logo_present | 43 | 43 | 2 | 0 | 0.96 | 1.00 |
| face_present | 22 | 22 | 0 | 0 | 1.00 | 1.00 |
| *source escalated to human* (expected = severe blur, text-heavy or logo missing) | 24 | 21 | 11 | 3 | 0.66 | 0.88 |

Face box IoU vs ground truth (detected faces): 0.91 · logo box IoU: 0.96

## Reframing: is everything important kept and clear of platform UI? (judged on ground-truth boxes)

| placement | centre crop ok | agent ok | agent auto-accepted | ok when accepted | false accepts | headline legible | mean steps | max steps |
|---|---|---|---|---|---|---|---|---|
| stories_9x16 | 2% | 100% | 45% | 100% | 0 | 100% | 2.77 | 3 |
| reels_9x16 | 2% | 100% | 43% | 100% | 0 | 95% | 2.77 | 3 |
| feed_1x1 | 25% | 97% | 45% | 93% | 2 | 100% | 2.05 | 3 |
| youtube_16x9 | 15% | 97% | 40% | 92% | 2 | 92% | 2.62 | 3 |
| display_300x250 | 5% | 92% | 25% | 87% | 2 | 65% | 2.55 | 3 |

**Overall:** centre crop 10% ok → agent 97% ok; 181 of 300 outputs escalated to a human; 6 false accepts.
**Runtime per creative (all 5 placements, CPU):** mean 1.7s, median 1.7s, max 2.1s

## Failure cases (17)

- `c006_1x1_astronaut_none_absent.png` detection: blurry
- `c019_4x5_product_blur_severe_safe.png` detection: severely_blurry
- `c020_1x1_rocket_blur_mild_safe.png` detection: severely_blurry, dark
- `c022_4x5_astronaut_none_safe.png` false_accept: feed_1x1 GT failed ['subject'], agent saw []
- `c022_4x5_astronaut_none_safe.png` false_accept: display_300x250 GT failed ['subject'], agent saw []
- `c023_4x5_astronaut_blur_mild_absent.png` detection: severely_blurry
- `c029_1x1_product_none_safe.png` detection: blurry
- `c030_1x1_camera_blur_mild_safe.png` false_accept: youtube_16x9 GT failed ['subject'], agent saw []
- `c036_9x16_camera_blur_mild_edge.png` detection: blurry
- `c043_1x1_coffee_blur_mild_edge.png` detection: severely_blurry
- `c045_1x1_camera_none_safe.png` false_accept: youtube_16x9 GT failed ['subject'], agent saw []
- `c046_4x5_camera_none_safe.png` false_accept: feed_1x1 GT failed ['subject'], agent saw []
- `c046_4x5_camera_none_safe.png` false_accept: display_300x250 GT failed ['subject'], agent saw []
- `c048_1x1_astronaut_blur_mild_absent.png` detection: severely_blurry
- `c051_1x1_coffee_blur_mild_edge.png` detection: severely_blurry
- `c057_4x5_product_none_distractor.png` detection: logo_present
- `c058_9x16_astronaut_none_distractor.png` detection: logo_present
