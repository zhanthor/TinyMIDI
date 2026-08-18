#include "audio_provider.h"

#include <Arduino.h>
#include <PDM.h>
#include <math.h>
#include <string.h>

#include "model_params.h"

// ---------------------------------------------------------------------------
// Microphone gain. The log-frequency features are gain-invariant by
// construction, so this only has to put the signal comfortably above the noise
// floor and below clipping. 20 is the library default; 40-50 suits a guitar at
// ~30 cm. It does NOT need calibrating the way a linear-magnitude front end
// would.
// ---------------------------------------------------------------------------
static const int kPdmGain = 40;

namespace {

// Ring buffer sized well beyond one window so an occasional slow inference does
// not immediately corrupt the frame being assembled.
constexpr int kRingSamples = 4096;          // 256 ms at 16 kHz

volatile int16_t  g_ring[kRingSamples];
volatile uint32_t g_write = 0;              // total samples ever written
volatile uint32_t g_read  = 0;              // total samples consumed
volatile uint32_t g_overruns = 0;

int16_t g_isr_buf[512];
float   g_last_rms = 0.0f;
bool    g_primed = false;

void OnPdmData() {
  const int bytes = PDM.available();
  if (bytes <= 0) return;
  const int n = (bytes > (int)sizeof(g_isr_buf)) ? (int)sizeof(g_isr_buf) : bytes;
  PDM.read((void*)g_isr_buf, n);

  const int samples = n / 2;
  for (int i = 0; i < samples; i++) {
    g_ring[(g_write + i) % kRingSamples] = g_isr_buf[i];
  }
  g_write += samples;
}

}  // namespace

bool AudioProviderInit() {
  memset((void*)g_ring, 0, sizeof(g_ring));
  g_write = g_read = g_overruns = 0;
  g_primed = false;

  PDM.onReceive(OnPdmData);
  PDM.setBufferSize(sizeof(g_isr_buf));
  PDM.setGain(kPdmGain);

  if (!PDM.begin(1, kSampleRate)) return false;   // 1 channel, 16 kHz
  return true;
}

bool AudioProviderReadWindow(float* out, int hop) {
  if (hop <= 0 || hop > kWindowSamples) return false;

  // Wait until a full window exists (first call) or `hop` fresh samples have
  // arrived (subsequent calls).
  const uint32_t need = g_primed ? (g_read + hop) : (uint32_t)kWindowSamples;
  while (g_write < need) {
    yield();
  }

  noInterrupts();
  const uint32_t w = g_write;
  interrupts();

  // If we fell far enough behind that the oldest sample we still want has been
  // overwritten, abandon the backlog and take the newest window instead. This
  // is the behaviour that keeps latency bounded on a slow core.
  if (w - g_read > (uint32_t)kRingSamples) {
    g_overruns += (w - g_read) - kRingSamples;
    g_read = w - kWindowSamples;
  }

  // Copy the most recent kWindowSamples, oldest first.
  const uint32_t start = w - kWindowSamples;
  double acc = 0.0;
  for (int i = 0; i < kWindowSamples; i++) {
    noInterrupts();
    const int16_t s = g_ring[(start + i) % kRingSamples];
    interrupts();
    // int16 -> [-1, 1], matching tf.audio.decode_wav's PCM_16 convention.
    const float v = (float)s / 32768.0f;
    out[i] = v;
    acc += (double)v * (double)v;
  }
  g_last_rms = (float)sqrt(acc / kWindowSamples);

  g_read = w;          // consume everything up to now; backlog is discarded
  g_primed = true;
  return true;
}

float AudioProviderLastRms() { return g_last_rms; }

uint32_t AudioProviderOverruns() { return g_overruns; }
