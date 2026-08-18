#include "feature_provider.h"

#include <math.h>

#include "filterbank.h"
#include "model_params.h"

// ---------------------------------------------------------------------------
// CMSIS-DSP acceleration.
//
// The nRF52840 is a 64 MHz Cortex-M4F -- roughly 3.75x slower than the ESP32
// this pipeline was first built for. arm_rfft_fast_f32 is a real-input FFT
// (about half the work of the complex one) with ROM twiddle tables, and it also
// saves ~28 KB of RAM by removing the imaginary array and the tables below.
//
// Set to 1 AFTER installing "Arduino_CMSIS-DSP" from the Library Manager.
// Leave at 0 to use the portable implementation, which is what has been
// verified against the Python reference.
//
// Either way, the golden test at boot tells you whether the path you chose
// reproduces the notebook's features. Trust that, not this comment.
// ---------------------------------------------------------------------------
#define USE_CMSIS_DSP 0

#if USE_CMSIS_DSP
  #include <arm_math.h>
  static arm_rfft_fast_instance_f32 g_rfft;
  static float g_buf[kFFTSize];
  static float g_out[kFFTSize];
#else
  #include "fft.h"
  static float g_re[kFFTSize];
  static float g_im[kFFTSize];
#endif

namespace {

float g_window[kWindowSamples];
float g_mag[kNumFFTBins];
bool  g_ready = false;

}  // namespace

bool FeatureProviderInit() {
  if (kFFTSize < kWindowSamples) return false;
  if (kNumFFTBins != kFFTSize / 2 + 1) return false;
  if (kFeatureElements != kNumLogBins) return false;

  // PERIODIC Hann -- divide by N, not (N-1).
  for (int n = 0; n < kWindowSamples; n++) {
    g_window[n] =
        0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)kWindowSamples);
  }

  unsigned int total = 0;
  for (int i = 0; i < kNumLogBins; i++) {
    if (kFbStart[i] + kFbLength[i] > kNumFFTBins) return false;
    total += kFbLength[i];
  }
  if (total != kFbWeights_len) return false;

#if USE_CMSIS_DSP
  if (arm_rfft_fast_init_f32(&g_rfft, kFFTSize) != ARM_MATH_SUCCESS) return false;
#else
  if (kFFTSize > kFFTMaxSize) return false;
  if (!FFTInit(kFFTSize)) return false;
#endif

  g_ready = true;
  return true;
}

void FeatureProviderCompute(const float* audio, float* out) {
  if (!g_ready) return;

#if USE_CMSIS_DSP
  for (int n = 0; n < kWindowSamples; n++) g_buf[n] = audio[n] * g_window[n];
  for (int n = kWindowSamples; n < kFFTSize; n++) g_buf[n] = 0.0f;

  arm_rfft_fast_f32(&g_rfft, g_buf, g_out, 0);

  // CMSIS packs the real FFT as:
  //   g_out[0] = DC (real), g_out[1] = Nyquist (real),
  //   g_out[2k], g_out[2k+1] = re, im for bin k (1 <= k < N/2).
  g_mag[0] = fabsf(g_out[0]);
  g_mag[kNumFFTBins - 1] = fabsf(g_out[1]);
  for (int k = 1; k < kFFTSize / 2; k++) {
    const float re = g_out[2 * k], im = g_out[2 * k + 1];
    g_mag[k] = sqrtf(re * re + im * im);
  }
#else
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
#endif

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

  // ---- clamp relative to this clip's peak, then remove the mean -----------
  // This pair is what makes the features exactly gain-invariant, which is why
  // the PDM gain setting is not critical on this board either.
  const float floor_v = peak - kLogTopRange;
  float mean = 0.0f;
  for (int i = 0; i < kNumLogBins; i++) {
    if (out[i] < floor_v) out[i] = floor_v;
    mean += out[i];
  }
  mean /= (float)kNumLogBins;
  for (int i = 0; i < kNumLogBins; i++) out[i] -= mean;
}
