// VibroSAR ESP32 firmware: 2x ADXL345 (I2C) -> USB serial, VibroSAR vibration protocol v1.
//
// Wiring (3.3 V ONLY):
//   ESP32 3V3 -> VCC of both ADXL345      ESP32 GND -> GND of both
//   ESP32 GPIO21 (SDA) -> SDA of both     ESP32 GPIO22 (SCL) -> SCL of both
//   CS pin of both boards -> 3V3  (selects I2C mode)
//   Sensor "target"    : SDO/ALT-ADDRESS -> GND  => I2C address 0x53
//   Sensor "reference" : SDO/ALT-ADDRESS -> 3V3  => I2C address 0x1D
//
// Channels sent (order must match configs/default.yaml channel_names): [target Z, reference Z]
// ADXL345 full-resolution mode: ~3.9 mg/LSB, so 1 g ~ 256 counts (use scale=0.0039 -> g).
// Sampling is 400 Hz, matched to the sensor output data rate to avoid duplicate/missed samples.
// Timestamps are the *scheduled* sample times (uniform grid) in ESP32 micros().

#include <Wire.h>

static const uint8_t  ADDR_TARGET = 0x53;
static const uint8_t  ADDR_REF    = 0x1D;
static const uint16_t FS_HZ       = 400;
static const uint8_t  N_CH        = 2;
static const uint8_t  N_SAMP      = 20;                  // 50 ms per frame
static const uint32_t PERIOD_US   = 1000000UL / FS_HZ;   // 2500 us
static const uint32_t BAUD        = 921600;

static const uint8_t REG_BW_RATE = 0x2C, REG_POWER_CTL = 0x2D, REG_DATA_FORMAT = 0x31, REG_DATAZ0 = 0x36;

static int16_t  buf[N_SAMP][N_CH];
static uint8_t  nbuf = 0;
static uint16_t seq = 0;
static uint32_t frameT0 = 0, nextUs = 0;
static int16_t  lastZ[N_CH] = {0, 0};

static uint16_t crc16(const uint8_t *d, size_t n) {       // CRC-16/CCITT-FALSE
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

static bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readZ(uint8_t addr, int16_t &z) {
  Wire.beginTransmission(addr); Wire.write(REG_DATAZ0);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)2) != 2) return false;
  uint8_t lo = Wire.read(), hi = Wire.read();
  z = (int16_t)((hi << 8) | lo);
  return true;
}

static bool initAdxl(uint8_t addr) {
  return writeReg(addr, REG_BW_RATE, 0x0C)        // 400 Hz output data rate
      && writeReg(addr, REG_DATA_FORMAT, 0x08)    // FULL_RES, +-2 g
      && writeReg(addr, REG_POWER_CTL, 0x08);     // measurement mode
}

static void put16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF; }

static void sendFrame() {
  uint8_t out[2 + 11 + N_SAMP * N_CH * 2 + 2];
  out[0] = 0xA5; out[1] = 0x5A;
  out[2] = 1;                                   // protocol version
  put16(out + 3, seq++);
  put32(out + 5, frameT0);
  put16(out + 9, FS_HZ);
  out[11] = N_CH; out[12] = N_SAMP;
  memcpy(out + 13, buf, sizeof(buf));           // little-endian int16, sample-major
  size_t bodyEnd = 13 + sizeof(buf);
  put16(out + bodyEnd, crc16(out + 2, bodyEnd - 2));
  Serial.write(out, bodyEnd + 2);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(BAUD);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  if (!(initAdxl(ADDR_TARGET) && initAdxl(ADDR_REF))) {
    while (true) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(100); }  // fast blink = sensor init failed
  }
  nextUs = micros() + PERIOD_US;
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextUs) < 0) return;
  if ((int32_t)(now - nextUs) > 10 * (int32_t)PERIOD_US) nextUs = now;   // fell far behind: resync
  if (nbuf == 0) frameT0 = nextUs;
  nextUs += PERIOD_US;

  // On an I2C error keep the last good value (rare; visible as flat segments).
  readZ(ADDR_TARGET, lastZ[0]);
  readZ(ADDR_REF, lastZ[1]);
  buf[nbuf][0] = lastZ[0];
  buf[nbuf][1] = lastZ[1];

  if (++nbuf == N_SAMP) { sendFrame(); nbuf = 0; }
}
