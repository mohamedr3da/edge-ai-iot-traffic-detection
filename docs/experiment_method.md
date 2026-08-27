# Experiment Method

## Collection Procedure

The Arduino streamed real IMU JSON to the laptop over USB serial. The laptop read the latest valid sensor sample and embedded it in HTTP POST requests sent to the Raspberry Pi. The laptop controlled only the request timing/profile. The Pi logged every event, extracted traffic-window features and ran local model inference.

The collection command was originally planned for a larger run count, but the final analysed dataset contains 30 completed hardware runs. The evidence therefore reports 30 total runs rather than 30 runs per class.

## Traffic Profiles

- Normal: steady telemetry-like request timing with occasional pauses.
- Burst: temporary legitimate spikes with gaps and jitter.
- Attack-like: sustained repetitive local requests to the Pi only.

Traffic profiles were varied around class boundaries, but the final window-rate distributions remained strongly separated. A tuned rate-only baseline is therefore reported alongside the ML models.

## Split IDs

### Train
- `cw2_real_20260818_attack_like_005`
- `cw2_real_20260818_attack_like_011`
- `cw2_real_20260818_attack_like_013`
- `cw2_real_20260818_attack_like_014`
- `cw2_real_20260818_attack_like_020`
- `cw2_real_20260818_attack_like_024`
- `cw2_real_20260818_attack_like_025`
- `cw2_real_20260818_attack_like_026`
- `cw2_real_20260818_burst_004`
- `cw2_real_20260818_burst_005`
- `cw2_real_20260818_burst_012`
- `cw2_real_20260818_burst_015`
- `cw2_real_20260818_burst_019`
- `cw2_real_20260818_burst_020`
- `cw2_real_20260818_burst_021`
- `cw2_real_20260818_burst_025`
- `cw2_real_20260818_burst_028`
- `cw2_real_20260818_burst_029`
- `cw2_real_20260818_normal_001`
- `cw2_real_20260818_normal_002`
- `cw2_real_20260818_normal_004`
- `cw2_real_20260818_normal_006`
- `cw2_real_20260818_normal_008`
- `cw2_real_20260818_normal_019`

### Validation
- `cw2_real_20260818_attack_like_012`
- `cw2_real_20260818_burst_003`
- `cw2_real_20260818_normal_030`

### Test
- `cw2_real_20260818_attack_like_003`
- `cw2_real_20260818_burst_023`
- `cw2_real_20260818_normal_021`

## Model Decision

The four learned ML models each achieved `1.000` validation macro-F1. Gaussian NB was selected before final test evaluation because it is dependency-free, small, interpretable and already compatible with the Pi server. On the held-out test, Gaussian NB and logistic regression tied at `macro-F1=0.886`, while logistic regression had the higher grouped-CV mean. The test and grouped-CV results are reported as evaluation evidence rather than used to select the deployed model.
