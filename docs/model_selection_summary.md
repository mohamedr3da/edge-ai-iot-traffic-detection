# Model selection summary

Selected model: `custom_gaussian_nb`.

Selection policy: compare candidate models on the validation split, then prefer the simplest interpretable deployment when performance is within 0.02 macro-F1. The final test split is reserved for reporting after selection.

The fixed threshold baseline shows why the original hand-written thresholds were weak. A second tuned rate-only baseline gives a fairer comparison against `messages_per_sec` alone; Gaussian NB is still retained because it combines traffic timing, byte rate and sensor-window features while remaining small and dependency-free at inference time.
