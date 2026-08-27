# Limitations

- The dataset contains 30 real hardware runs; larger multi-client collection remains future work.
- The clean evaluated experiment has one traffic client, so it is local DoS-like / attack-like detection rather than distributed DDoS.
- The fixed grouped test set is small because one scenario per class went into the test split; grouped cross-validation is reported alongside it.
- The home Wi-Fi/router path is realistic for a small-site network, but it is not a controlled network laboratory.
- No wattmeter was available, so wall-power consumption is not measured.
- External CICIoT2023/N-BaIoT validation was not performed; benchmark datasets are used only for feature-mapping context.
- The model is a prototype classifier for controlled local traffic patterns, not a production IDS.
