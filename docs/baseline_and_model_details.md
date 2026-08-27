# Baseline And Model Details

## Baseline Rule

```text
if messages_per_sec >= 24 -> attack_like
else if messages_per_sec >= 16 and active_clients >= 2 -> attack_like
else if messages_per_sec >= 8 or std_interarrival_ms > 120 -> burst
else -> normal
```

The fixed baseline failed because its hand-written thresholds did not match the observed one-client traffic ranges. It never recognised attack-like test windows.

- Fixed grouped test macro-F1: `0.222`
- Fixed grouped test attack-like recall: `0.000`
- Grouped CV macro-F1: `0.302 +/- 0.044`

## Tuned Rate-Only Baseline

A second baseline uses only `messages_per_sec`. Two thresholds were selected from the validation split:

```text
if messages_per_sec < 5.75 -> normal
else if messages_per_sec < 9.65 -> burst
else -> attack_like
```

This baseline is included because it is a stronger comparator than the old fixed rule.

- Fixed grouped test macro-F1: `0.750`
- Fixed grouped test attack-like recall: `1.000`
- Fixed grouped test burst false alarm rate: `0.000`
- Evidence file: `evaluation/metrics/tuned_rate_baseline.json`

## Gaussian NB Implementation

- Implementation: custom dependency-free Python class in `pi_edge_ai/ml_model.py`.
- Features: raw numeric feature values, no standardisation.
- Learned parameters: class priors, per-class means, per-class variances.
- Variance floor: `1e-6`.
- Prediction: Gaussian log likelihood plus log prior, converted to probabilities using softmax.
- Saved model: JSON, `2329` bytes for the final selected export.

## Final Gaussian NB Results

- Fixed grouped test macro-F1: `0.886`
- Fixed grouped test attack-like recall: `1.000`
- Fixed grouped test burst false alarm rate: `0.000`
- Grouped 5-fold CV macro-F1: `0.952 +/- 0.026`
