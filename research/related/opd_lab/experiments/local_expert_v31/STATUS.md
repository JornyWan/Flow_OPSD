# Phase31 complete: isolated learned actions reduce some harms but add no complete scenes

One frozen route: S1 prefix3 / verified S2 actions2 / S1 suffix5, same36 exposed Phase29 development pairs. No training, checkpoint selection, window search or teacher search. Three references are inherited verbatim (108 observations);36 new native512 images were observed before joining keys or action-error aggregates. Review remains provisional implementing-assistant observation, not independent human confirmation.

| Route | Complete /36 | Text | Task | Visibility | Unknown | Known harm vs S1 |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 16 | 19 | 29 | 29 | 4 | 0 |
| S1 / OCR2 / S1 | 18 | 22 | 31 | 32 | 2 | 1 |
| Full verified S2 | 16 | 23 | 27 | 30 | 4 | 6 |
| S1 / local S2 / S1 | 16 | 21 | 27 | 29 | 6 | 1 |

Local minus S1:0pp,95% paired prompt-cluster CI[-11.11,+13.89]. Local minus full S2:0pp[-16.67,+13.89]. Local minus OCR2:-5.56pp[-16.67,+5.56]. All uncertainty sensitivity intervals are in results.json and report.html. No net gain is established.

All six full-S2 harm cases were examined categorically: cases8,9,17,29,34 regain every previously harmed field with local S2; case25 still violates the book-count/relation clause. No previously harmed field becomes unknown. This is a posthoc descriptive decomposition, not a performance estimate on a selected subset. The intervention changes prefix and suffix jointly; it does not identify their separate effects.

On72 identical saved development query states, S1-to-OCR action MSE0.1620489475, S2-to-OCR0.1445148069:10.8203% reduction. All states/targets/predictions saved; CPU reconstruction agrees. This is action fitting, not an image-quality metric.

Job1059794 completed86 allocated seconds,0.0238888889 GPU-hours. Three exact state/action/PNG replays (S1,OCR2,full S2),360 saved Euler transitions,252 new generation transitions,144 same-state action predictions,425 CFG-expanded forward examples including replays,39 decodes,zero backward/optimizer steps. All four scores,six paired intervals,144 embedded image hashes and108 inherited labels verified. Final test ungenerated.
