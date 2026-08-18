#include "ble_midi.h"

#include <Arduino.h>
#include <ArduinoBLE.h>

const char* kDeviceName = "TinyMIDI-Guitar";

namespace {

// Standard MIDI-over-BLE service and characteristic UUIDs. These are fixed by
// the BLE MIDI spec -- hosts scan for exactly these.
BLEService midiService("03B80E5A-EDE8-4B33-A751-6CE34EC4C700");
BLECharacteristic midiChar("7772E5DB-3868-4112-A1A9-F2669D106BF3",
                           BLERead | BLEWriteWithoutResponse | BLENotify, 5);

bool g_ready = false;

// A BLE MIDI packet is [header, timestamp, status, data1, data2].
// Both header and timestamp carry the high bit set; the remaining 13 bits are a
// millisecond counter split 6/7. Hosts use it to de-jitter, so it must advance.
void SendMidi(uint8_t status, uint8_t d1, uint8_t d2) {
  if (!g_ready) return;

  const uint16_t ts = (uint16_t)(millis() & 0x1FFF);
  uint8_t pkt[5];
  pkt[0] = 0x80 | ((ts >> 7) & 0x3F);
  pkt[1] = 0x80 | (ts & 0x7F);
  pkt[2] = status;
  pkt[3] = d1 & 0x7F;
  pkt[4] = d2 & 0x7F;

  midiChar.writeValue(pkt, sizeof(pkt));
}

}  // namespace

bool BleMidiInit() {
  if (!BLE.begin()) return false;

  BLE.setLocalName(kDeviceName);
  BLE.setDeviceName(kDeviceName);
  midiService.addCharacteristic(midiChar);
  BLE.addService(midiService);
  BLE.setAdvertisedService(midiService);
  BLE.advertise();

  g_ready = true;
  return true;
}

void BleMidiPoll() {
  if (g_ready) BLE.poll();
}

bool BleMidiConnected() {
  return g_ready && BLE.connected();
}

void BleMidiNoteOn(int note, int velocity) {
  if (note < 0 || note > 127) return;
  if (velocity < 1) velocity = 1;
  if (velocity > 127) velocity = 127;
  SendMidi(0x90, (uint8_t)note, (uint8_t)velocity);   // note on, channel 1
}

void BleMidiNoteOff(int note) {
  if (note < 0 || note > 127) return;
  SendMidi(0x80, (uint8_t)note, 0);                   // note off, channel 1
}
