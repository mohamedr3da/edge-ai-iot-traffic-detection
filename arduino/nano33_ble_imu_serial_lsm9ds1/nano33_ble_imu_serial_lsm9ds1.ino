#include <Arduino_LSM9DS1.h>

const unsigned long SAMPLE_INTERVAL_MS = 40; // 25 Hz
unsigned long lastSampleMs = 0;
unsigned long seq = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  if (!IMU.begin()) {
    Serial.println("{\"error\":\"imu_begin_failed\"}");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("{\"status\":\"ready\",\"board\":\"nano33_ble_lsm9ds1\"}");
}

void loop() {
  const unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }

  Serial.print("{\"device\":\"nano33ble\",\"seq\":");
  Serial.print(seq++);
  Serial.print(",\"t_ms\":");
  Serial.print(now);
  Serial.print(",\"ax\":");
  Serial.print(ax, 5);
  Serial.print(",\"ay\":");
  Serial.print(ay, 5);
  Serial.print(",\"az\":");
  Serial.print(az, 5);
  Serial.print(",\"gx\":");
  Serial.print(gx, 5);
  Serial.print(",\"gy\":");
  Serial.print(gy, 5);
  Serial.print(",\"gz\":");
  Serial.print(gz, 5);
  Serial.println("}");
}

