#include "audio_provider.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "model_params.h"

// ---------------------------------------------------------------------------
// Pin map -- edit to match your wiring.
// ---------------------------------------------------------------------------
#define I2S_PIN_BCLK 14
#define I2S_PIN_WS   15
#define I2S_PIN_DIN  32

// ---------------------------------------------------------------------------
// Microphone gain.
//
// The INMP441 is a 24-bit part; we normalize by 2^31 so a full-scale acoustic
// input maps to +/-1.0, matching tf.audio.decode_wav. In practice a MEMS mic
// at conversational distance sits far below full scale, so this multiplier
// brings the level into the range the model was trained on.
//
// START AT 1.0, run the sketch, and watch the RMS figure printed by the
// serial output while you play a note. Aim for RMS in the 0.02-0.20 band --
// roughly where the training WAVs sit. Then set kMicGain accordingly.
// ---------------------------------------------------------------------------
float kMicGain = 8.0f;

// ---------------------------------------------------------------------------
// The Arduino ESP32 core changed its I2S API in v3.x (ESP-IDF 5.x). Both
// paths are here so this compiles either way.
// ---------------------------------------------------------------------------
#include "esp_arduino_version.h"

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #include "driver/i2s_std.h"
  static i2s_chan_handle_t g_rx_handle = nullptr;
#else
  #include "driver/i2s.h"
  static const i2s_port_t kI2SPort = I2S_NUM_0;
#endif

namespace {

constexpr int kDmaFrames  = 256;
constexpr int kDmaBuffers = 8;      // 8 * 256 / 16000 = 128 ms of slack

float   g_window[kWindowSamples];
int32_t g_raw[kWindowSamples];      // scratch for one hop of I2S words
float   g_last_rms = 0.0f;
bool    g_primed   = false;

}  // namespace

bool AudioProviderInit() {
  memset(g_window, 0, sizeof(g_window));

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  i2s_chan_config_t chan_cfg =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num  = kDmaBuffers;
  chan_cfg.dma_frame_num = kDmaFrames;
  if (i2s_new_channel(&chan_cfg, nullptr, &g_rx_handle) != ESP_OK) return false;

  i2s_std_config_t std_cfg = {
      .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                      I2S_SLOT_MODE_MONO),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = (gpio_num_t)I2S_PIN_BCLK,
          .ws   = (gpio_num_t)I2S_PIN_WS,
          .dout = I2S_GPIO_UNUSED,
          .din  = (gpio_num_t)I2S_PIN_DIN,
          .invert_flags = {false, false, false},
      },
  };
  // L/R tied to GND -> the mic transmits in the left slot.
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

  if (i2s_channel_init_std_mode(g_rx_handle, &std_cfg) != ESP_OK) return false;
  if (i2s_channel_enable(g_rx_handle) != ESP_OK) return false;

#else
  i2s_config_t cfg = {};
  cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate          = kSampleRate;
  cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count        = kDmaBuffers;
  cfg.dma_buf_len          = kDmaFrames;
  cfg.use_apll             = false;
  cfg.tx_desc_auto_clear   = false;
  cfg.fixed_mclk           = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num   = I2S_PIN_BCLK;
  pins.ws_io_num    = I2S_PIN_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = I2S_PIN_DIN;

  if (i2s_driver_install(kI2SPort, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(kI2SPort, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(kI2SPort);
#endif

  g_primed = false;
  return true;
}

bool AudioProviderReadWindow(float* out, int hop) {
  if (hop <= 0 || hop > kWindowSamples) return false;

  // On the very first call, fill the whole window so we do not classify a
  // buffer that is mostly zeros.
  const int want = g_primed ? hop : kWindowSamples;

  int filled = 0;
  while (filled < want) {
    const int chunk = want - filled;
    size_t bytes_read = 0;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
    if (i2s_channel_read(g_rx_handle, g_raw + filled, chunk * sizeof(int32_t),
                         &bytes_read, portMAX_DELAY) != ESP_OK) {
      return false;
    }
#else
    if (i2s_read(kI2SPort, g_raw + filled, chunk * sizeof(int32_t),
                 &bytes_read, portMAX_DELAY) != ESP_OK) {
      return false;
    }
#endif
    if (bytes_read == 0) return false;
    filled += (int)(bytes_read / sizeof(int32_t));
  }

  // Slide the window left by `want` and append the new samples.
  if (want < kWindowSamples) {
    memmove(g_window, g_window + want,
            (kWindowSamples - want) * sizeof(float));
  }

  float* tail = g_window + (kWindowSamples - want);
  for (int i = 0; i < want; i++) {
    // The INMP441 places its 24-bit sample in the upper bits of the 32-bit
    // slot. Dividing by 2^31 normalizes to [-1, 1] in one step and keeps the
    // low bits, which matters because we are looking at low-level harmonics.
    tail[i] = ((float)g_raw[i] / 2147483648.0f) * kMicGain;
  }

  g_primed = true;

  double acc = 0.0;
  for (int i = 0; i < kWindowSamples; i++) {
    acc += (double)g_window[i] * (double)g_window[i];
  }
  g_last_rms = (float)sqrt(acc / kWindowSamples);

  memcpy(out, g_window, kWindowSamples * sizeof(float));
  return true;
}

float AudioProviderLastRms() { return g_last_rms; }
