# Joint v2 baseline data views

train.txt / train.csv contain the same512 training prompts in the same order.
Native `test.txt` is explicitly the128 development prompts, not confirmation or
final-test data. development.csv is identical. Native text readers return empty
metadata; the exact joint constraints are preserved in separate sidecars.
NFT still needs an explicit joint-reward adapter and cost audit before a formal
baseline run. Exporting text is not implementing that reward or reproducing NFT.
No confirmation/final prompts are included. No model or reward was queried.
