// ble_midi.h -- BLE MIDI output over the standard MIDI-over-BLE service.
//
// This turns the board into an actual wireless MIDI controller: play the
// guitar, and a soft-synth on a laptop or phone plays the detected notes.
//
// Pairing:
//   macOS  -- Audio MIDI Setup -> Window -> Show MIDI Studio -> Bluetooth
//   iOS    -- any BLE-MIDI host app (GarageBand via a BLE MIDI connector app)
//   Windows-- MIDIberry / loopMIDI + a BLE MIDI bridge
//   Linux  -- bluez + the ble-midi ALSA sequencer bridge
//
// The presentation guidelines warn that many BLE devices will be active in the
// room, so kDeviceName is deliberately distinctive -- change it to something
// unique to your team and verify pairing before the session.

#pragma once

#include <stdint.h>

// Advertised BLE name. MAKE THIS UNIQUE before your demo.
extern const char* kDeviceName;

// Returns false if the BLE stack failed to start.
bool BleMidiInit();

// Must be called often from loop() to service the BLE stack.
void BleMidiPoll();

bool BleMidiConnected();

// note: 0-127. velocity: 1-127.
void BleMidiNoteOn(int note, int velocity);
void BleMidiNoteOff(int note);
