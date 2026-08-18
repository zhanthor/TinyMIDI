// audio_provider.h -- I2S capture for INMP441 / SPH0645 on ESP32.
//
// Replaces Lab 5's arduino_audio_provider.cpp, which is built on the Nano 33
// BLE's PDM library and does not exist on Xtensa.
//
// ---- Wiring (INMP441) ----------------------------------------------------
//   INMP441        ESP32
//   -------        -----
//   VDD            3V3        (NOT 5V)
//   GND            GND
//   SCK  (BCLK)    GPIO14
//   WS   (LRCL)    GPIO15
//   SD   (DOUT)    GPIO32
//   L/R            GND        -> mic drives the LEFT slot
//
// Change the pins in audio_provider.cpp if you wired it differently.

#pragma once

// Starts the I2S peripheral. Returns false on driver failure.
bool AudioProviderInit();

// Blocks until `hop` new samples have arrived, then slides the internal
// window and copies the most recent kWindowSamples into `out` as floats
// scaled to match tf.audio.decode_wav's [-1, 1] convention (with kMicGain
// applied). Returns false on an I2S read error.
bool AudioProviderReadWindow(float* out, int hop);

// RMS of the window most recently returned. Cheap energy gate input.
float AudioProviderLastRms();
