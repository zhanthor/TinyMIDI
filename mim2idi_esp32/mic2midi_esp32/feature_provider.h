// feature_provider.h -- log-frequency feature extraction.
//
// This file has no equivalent in Lab 5. micro_speech's
// micro_features_generator.cpp produces 40-bin log-Mel microfrontend features
// over a 1-second window; this model wants a 29x65-bin log-FREQUENCY (musical,
// 3 bins per semitone) representation over a 64 ms window. Mel spacing is
// tuned to speech perception and is far too coarse in the 80-500 Hz band
// where these note fundamentals live.
//
// Reference behaviour being matched (see notebook section 4):
//
//   spec = |rfft(audio * hann(1024, periodic), n=2048)|     -> 1025 bins
//   e    = FILTERBANK @ spec                                 ->  29x65 bins
//   f    = log(e + 1e-6)
//   out  = f - mean(f)
//
// The final mean subtraction makes the output exactly invariant to a scalar
// gain on the input, which is what removes the microphone-calibration
// problem the previous linear-magnitude front end had.

#pragma once

// Call once from setup(). Builds the Hann window and FFT tables, and
// validates the generated filterbank against model_params.h.
// Returns false on any geometry mismatch.
bool FeatureProviderInit();

// Computes features for `audio` (kWindowSamples floats) into `out`
// (kFeatureElements == kNumLogBins floats).
void FeatureProviderCompute(const float* audio, float* out);
