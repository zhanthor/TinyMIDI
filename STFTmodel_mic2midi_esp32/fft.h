// fft.h -- minimal radix-2 complex FFT, in-place.
//
// Deliberately dependency-free. ESP-DSP's dsps_fft2r_fc32 is faster, but it
// is an ESP-IDF component and is awkward to pull into an Arduino IDE sketch.
// At 128 points x 29 frames this costs well under a millisecond on a 240 MHz
// ESP32 with the hardware FPU, which is nothing next to the ~10 ms inference.

#pragma once

#include <stdint.h>

// Must be called once before FFTTransform. `n` must be a power of two.
// Returns false if n is not a supported power of two or exceeds kFFTMaxSize.
bool FFTInit(int n);

// In-place decimation-in-time FFT. `re` and `im` are each `n` floats.
// Pass an all-zero `im` for real input.
void FFTTransform(float* re, float* im);

constexpr int kFFTMaxSize = 2048;
