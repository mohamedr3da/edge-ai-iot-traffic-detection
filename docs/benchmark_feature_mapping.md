# Benchmark feature mapping

CICIoT2023 and N-BaIoT are used as benchmark context, not as external validation for this project.

| Project feature | Benchmark overlap | Comment |
|---|---|---|
| messages_per_sec, message_count, bytes_per_sec | Partial | Comparable to flow/count/rate-derived network features after preprocessing. |
| mean/std inter-arrival, interarrival_cv | Partial | Comparable in principle if packet timestamps or flow durations are available. |
| active_clients | Partial | Requires source grouping; direct IP shortcuts must be avoided. |
| sensor_accel_mag_mean/std, sensor_gyro_mag_mean | No direct overlap | Specific to this Arduino sensor-node experiment. |

Direct validation would require a preprocessing adapter that rebuilds the same configured feature-window definitions from each benchmark. That was not claimed here.
