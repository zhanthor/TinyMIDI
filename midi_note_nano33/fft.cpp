#include "fft.h"

#include <math.h>

namespace {

int      g_n = 0;
int      g_levels = 0;
float    g_cos[kFFTMaxSize / 2];
float    g_sin[kFFTMaxSize / 2];
uint16_t g_rev[kFFTMaxSize];

}  // namespace

bool FFTInit(int n) {
  if (n < 2 || n > kFFTMaxSize) return false;
  if (n & (n - 1)) return false;  // not a power of two

  g_n = n;
  g_levels = 0;
  for (int t = n; t > 1; t >>= 1) g_levels++;

  for (int i = 0; i < n / 2; i++) {
    const float angle = -2.0f * (float)M_PI * (float)i / (float)n;
    g_cos[i] = cosf(angle);
    g_sin[i] = sinf(angle);
  }

  // Bit-reversal permutation table.
  for (int i = 0; i < n; i++) {
    int x = i;
    int r = 0;
    for (int b = 0; b < g_levels; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    g_rev[i] = (uint16_t)r;
  }
  return true;
}

void FFTTransform(float* re, float* im) {
  const int n = g_n;

  // Reorder into bit-reversed index order.
  for (int i = 0; i < n; i++) {
    const int j = g_rev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i];       im[i] = im[j]; im[j] = t;
    }
  }

  // Cooley-Tukey butterflies.
  for (int size = 2; size <= n; size <<= 1) {
    const int half = size >> 1;
    const int step = n / size;
    for (int i = 0; i < n; i += size) {
      for (int j = i, k = 0; j < i + half; j++, k += step) {
        const float c = g_cos[k];
        const float s = g_sin[k];
        const int   m = j + half;

        const float tre = re[m] * c - im[m] * s;
        const float tim = re[m] * s + im[m] * c;

        re[m] = re[j] - tre;
        im[m] = im[j] - tim;
        re[j] += tre;
        im[j] += tim;
      }
    }
  }
}
