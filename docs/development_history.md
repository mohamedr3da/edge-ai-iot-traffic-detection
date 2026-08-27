# Development history evidence

An early pre-final attack-like pilot run was discarded before the final 30-run collection because blocking Arduino serial reads throttled the laptop traffic generator. The gateway was updated to keep the last valid IMU sample and use short non-blocking serial reads, allowing HTTP request timing to remain controlled.

This matters because the experiment compares traffic patterns. A run where the serial reader accidentally limits request rate is not valid evidence of attack-like sustained pressure.

No runs were excluded from the final analysed 30-run dataset.
