# Hardware, BOM And Power Notes

## Tested Hardware

| Item | Role | Evidence |
|---|---|---|
| Arduino Nano 33 BLE/Sense-compatible board | Live IMU sensor node over USB serial | Windows detected `USB Serial Device (COM4)` with Arduino USB VID/PID; `evidence/arduino_serial_samples.jsonl` contains real IMU JSON. |
| Windows laptop | Serial receiver and controlled local request gateway | Laptop IP in Pi logs: `192.168.1.207`; environment captured in `evidence/windows_laptop_info.txt`. |
| Normal local Wi-Fi/router | Local network path between laptop and Pi | Pi IP `192.168.1.187`; the Pi was only the edge inference server. |
| Raspberry Pi 5 Model B Rev 1.1 | Edge inference server/dashboard | `evidence/pi_system_info.txt`; measured 2.0 GiB RAM in this unit. |
| microSD card, USB cable and USB-C PSU | Storage, Arduino link and Pi power | Required support items used to run the local hardware testbed. |

## Specification Sources

- Raspberry Pi 5 official product page: https://www.raspberrypi.com/products/raspberry-pi-5/
- Arduino Nano 33 BLE Sense official documentation: https://docs.arduino.cc/hardware/nano-33-ble-sense/

## Power

No wall-power measurement was taken because no wattmeter or USB-C power meter was available.

Raspberry Pi 5 power is treated as a hardware requirement rather than a measured experimental metric. The official Raspberry Pi specification lists 5V/5A DC over USB-C with USB Power Delivery support and recommends a high-quality 5V/5A USB-C supply, such as the official 27W unit. The experiment measured compute resource use instead: CPU percentage and RSS memory during traffic collection.

## Resource Use Measured

Resource monitoring during the final collection is in `evaluation/resource_tests/resource_usage_summary.json` and `resource_usage_timeseries.png`.
