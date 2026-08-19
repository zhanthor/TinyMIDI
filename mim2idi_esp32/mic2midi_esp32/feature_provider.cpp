#include "feature_provider.h"

#include <math.h>

#include "fft.h"
#include "model_params.h"

namespace {

float g_window[kFrameLength];
float g_re[kFrameLength];
float g_im[kFrameLength];

bool g_ready = false;

}

bool FeatureProviderInit() {
  if (!FFTInit(kFrameLength)) {
    return false;
  }

  for (int n = 0; n < kFrameLength; ++n) {
    g_window[n] =
        0.5f -
        0.5f * cosf(2.0f * (float)M_PI *
                    (float)n / (float)kFrameLength);
  }

  g_ready = true;
  return true;
}

void FeatureProviderCompute(const float* audio, float* out) {
  if (!g_ready) return;

  for (int frame = 0; frame < kNumFrames; ++frame) {
    const int start = frame * kFrameStep;

    for (int n = 0; n < kFrameLength; ++n) {
      g_re[n] = audio[start + n] * g_window[n];
      g_im[n] = 0.0f;
    }

    FFTTransform(g_re, g_im);

    for (int k = 0; k < kNumFFTBins; ++k) {
      out[frame * kNumFFTBins + k] =
          sqrtf(g_re[k] * g_re[k] +
                g_im[k] * g_im[k]);
    }
  }
}