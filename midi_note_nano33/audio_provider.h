// audio_provider.h -- PDM microphone capture for the Arduino Nano 33 BLE Sense.
//
// IMPORTANT: the plain "Nano 33 BLE" has NO microphone. Only the "Nano 33 BLE
// Sense" (rev 1 or rev 2) carries the MP34DT05/MP34DT06 PDM mic. On a non-Sense
// board this file compiles but captures nothing.
//
// Differences from the ESP32 version, and why they matter here:
//
//   * The PDM library delivers samples from an interrupt, so capture is
//     decoupled from the main loop. On a 64 MHz M4 the inference can take
//     longer than the hop, so the ring buffer WILL overrun sometimes.
//   * On overrun we drop the backlog and keep the newest audio rather than
//     queueing it. A pitch detector that reports notes from three seconds ago
//     is worse than one that skips frames.

#pragma once

#include <stdint.h>

// Starts the PDM microphone at kSampleRate, mono. Returns false on failure.
bool AudioProviderInit();

// Blocks until at least `hop` new samples have arrived, then writes the most
// recent kWindowSamples into `out` as floats in [-1, 1] (matching
// tf.audio.decode_wav). Returns false only on a hard failure.
bool AudioProviderReadWindow(float* out, int hop);

// RMS of the window most recently returned.
float AudioProviderLastRms();

// Number of samples dropped because the main loop could not keep up. Non-zero
// means the hop is shorter than the processing time -- raise kHopSamples.
uint32_t AudioProviderOverruns();
