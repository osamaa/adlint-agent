# AdLint evaluation (llm planner, 20 creatives x 5 placements)

## Detection (source creative)

| check | positives | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| blurry | 4 | 4 | 1 | 0 | 0.80 | 1.00 |
| severely_blurry | 2 | 1 | 0 | 1 | 1.00 | 0.50 |
| dark | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| text_heavy | 2 | 2 | 0 | 0 | 1.00 | 1.00 |
| logo_present | 14 | 14 | 0 | 0 | 1.00 | 1.00 |
| face_present | 6 | 6 | 0 | 0 | 1.00 | 1.00 |
| *source escalated to human* (expected = severe blur, text-heavy or logo missing) | 9 | 8 | 4 | 1 | 0.67 | 0.89 |

Face box IoU vs ground truth (detected faces): 0.91 · logo box IoU: 0.97

## Reframing: is everything important kept and clear of platform UI? (judged on ground-truth boxes)

| placement | centre crop ok | agent ok | agent auto-accepted | ok when accepted | false accepts | headline legible | mean steps | max steps |
|---|---|---|---|---|---|---|---|---|
| stories_9x16 | 5% | 100% | 40% | 100% | 0 | 100% | 1.95 | 2 |
| reels_9x16 | 5% | 100% | 40% | 100% | 0 | 95% | 1.95 | 2 |
| feed_1x1 | 15% | 100% | 40% | 100% | 0 | 100% | 1.75 | 4 |
| youtube_16x9 | 20% | 100% | 35% | 100% | 0 | 95% | 2.15 | 3 |
| display_300x250 | 0% | 95% | 15% | 100% | 0 | 65% | 2.15 | 3 |

**Overall:** centre crop 9% ok → agent 99% ok; 66 of 100 outputs escalated to a human; 0 false accepts.
**Runtime per creative (all 5 placements, CPU):** mean 20.1s, median 19.8s, max 27.8s

## Failure cases (2)

- `c006_1x1_astronaut_none_absent.png` detection: blurry
- `c019_4x5_product_blur_severe_safe.png` detection: severely_blurry
