#include "feature_provider.h"

#include <math.h>

#include "fft.h"
#include "filterbank.h"
#include "model_params.h"

// Replicates the notebook's get_features() exactly:
//
//   1024 samples -> periodic Hann -> rfft(n=2048) -> magnitude   (1025 bins)
//                -> sparse log-frequency triangular bank          (217 bins)
//                -> log(x + eps)
//                -> clamp at (max - kLogTopRange)
//                -> subtract per-clip mean
//
// The mean subtraction is what buys gain invariance: a scalar gain g on the
// input becomes an additive log(g) after the log, which the mean removes
// exactly. That is why kMicGain is no longer critical.

namespace {

float g_window[kWindowSamples];
float g_re[kFFTSize];
float g_im[kFFTSize];
float g_mag[kNumFFTBins];

bool g_ready = false;

}  // namespace

bool FeatureProviderInit() {
  if (kFFTSize > kFFTMaxSize) return false;
  if (kFFTSize < kWindowSamples) return false;
  if (kNumFFTBins != kFFTSize / 2 + 1) return false;
  if (kFeatureElements != kNumLogBins) return false;

  // PERIODIC Hann -- divide by N, not (N-1). tf.signal.hann_window and
  // np.hanning(N+1)[:N] both use the periodic form. The symmetric version is
  // a silent mismatch: it runs fine, the numbers are just wrong.
  for (int n = 0; n < kWindowSamples; n++) {
    g_window[n] =
        0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)kWindowSamples);
  }

  // Verify the sparse filterbank tables agree with the declared geometry.
  unsigned int total = 0;
  for (int i = 0; i < kNumLogBins; i++) {
    if (kFbStart[i] + kFbLength[i] > kNumFFTBins) return false;
    total += kFbLength[i];
  }
  if (total != kFbWeights_len) return false;

  if (!FFTInit(kFFTSize)) return false;

  g_ready = true;
  return true;
}

void FeatureProviderCompute(const float* audio, float* out) {
  if (!g_ready) return;

  // ---- windowed, zero-padded FFT over the whole 64 ms clip ----------------
  for (int n = 0; n < kWindowSamples; n++) {
    g_re[n] = audio[n] * g_window[n];
    g_im[n] = 0.0f;
  }
  for (int n = kWindowSamples; n < kFFTSize; n++) {
    g_re[n] = 0.0f;
    g_im[n] = 0.0f;
  }

  FFTTransform(g_re, g_im);

  for (int k = 0; k < kNumFFTBins; k++) {
    g_mag[k] = sqrtf(g_re[k] * g_re[k] + g_im[k] * g_im[k]);
  }

  // ---- sparse triangular filterbank + log ---------------------------------
  const float* w = kFbWeights;
  float peak = -1e30f;
  for (int i = 0; i < kNumLogBins; i++) {
    const float* m = g_mag + kFbStart[i];
    const int    n = kFbLength[i];

    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += w[j] * m[j];
    w += n;

    const float v = logf(acc + kLogEps);
    out[i] = v;
    if (v > peak) peak = v;
  }

  // ---- clamp relative to this clip's peak ---------------------------------
  // Bins with no signal energy land near the numerical noise floor, where
  // logf() is unstable and the value means nothing. Flooring relative to the
  // peak (rather than at an absolute epsilon) keeps the whole pipeline exactly
  // gain-invariant, since the floor shifts by log(g) along with everything
  // else and the mean subtraction below cancels it.
  const float floor_v = peak - kLogTopRange;
  float mean = 0.0f;
  for (int i = 0; i < kNumLogBins; i++) {
    if (out[i] < floor_v) out[i] = floor_v;
    mean += out[i];
  }

  // ---- per-clip mean subtraction (this is the gain invariance) ------------
  mean /= (float)kNumLogBins;
  for (int i = 0; i < kNumLogBins; i++) out[i] -= mean;
}
