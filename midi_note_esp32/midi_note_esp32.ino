// =============================================================================
//  MIDI note classifier on ESP32
//  EE 446 final project -- deployment via the Lab 5 workflow
// =============================================================================
//
//  Pipeline:  I2S mic -> 128 ms window -> 2048-point FFT
//             -> log-frequency filterbank (217 bins, 3/semitone)
//             -> log + max-relative floor + mean subtraction
//             -> int8 quantize -> TFLM interpreter -> softmax
//             -> posterior smoothing -> serial
//
//  The feature front end is gain-invariant by construction, so microphone
//  level calibration is a sanity check rather than a requirement.
//
//  Required library (Arduino IDE -> Library Manager): "tflm_esp32"
//  This is Espressif's esp-tflite-micro packaged for Arduino. The
//  Arduino_TensorFlowLite library used in Lab 5 is Cortex-M only and will not
//  compile for Xtensa.
//
//  Board: Tools -> Board -> ESP32 Dev Module
//         Tools -> Partition Scheme -> "Huge APP (3MB No OTA/1MB SPIFFS)"
//         The model is ~230 KB, which pushes past the default partition once
//         the TFLM runtime is linked in.
//
//  Generate model.h, model_params.h, filterbank.h and golden_test.h from the
//  notebook (section 9) and drop them in this folder before compiling.
// =============================================================================

#include <tflm_esp32.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_provider.h"
#include "feature_provider.h"
#include "model.h"
#include "model_params.h"
#include "note_recognizer.h"

// Comment out once you trust the port.
#define ENABLE_GOLDEN_TEST
#ifdef ENABLE_GOLDEN_TEST
#include "golden_test.h"
#endif

// -----------------------------------------------------------------------------
// Tuning
// -----------------------------------------------------------------------------

// How far the 128 ms analysis window advances per inference. 512 samples =
// 32 ms, so consecutive windows overlap 75% and you get ~31 inferences/sec.
//
// Note that the hop sets the UPDATE rate, not the detection latency: a newly
// struck note is not fully inside the window until 128 ms have passed. That is
// the price of the frequency resolution -- see the notebook intro.
constexpr int kHopSamples = 512;

// Skip inference entirely below this RMS. The model has a `silence` class so
// this is belt-and-braces, but it keeps the CPU idle in a quiet room and
// stops the recognizer's history filling with noise-driven garbage.
constexpr float kRmsGate = 0.004f;

// TFLM scratch memory. The log-frequency model's largest activation is only
// 217*24 bytes, so this is generous. The boot printout reports actual usage.
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

namespace {

const tflite::Model*     g_model_ptr  = nullptr;
tflite::MicroInterpreter* g_interp    = nullptr;
TfLiteTensor*            g_input      = nullptr;
TfLiteTensor*            g_output     = nullptr;

float g_audio[kWindowSamples];
float g_features[kFeatureElements];
float g_scores[kCategoryCount];

// Frame-counted. At a 32 ms hop, 6 frames is ~192 ms of smoothing.
NoteRecognizer g_recognizer(/*history_frames=*/6, /*detection_thresh=*/0.55f,
                            /*suppression_ms=*/250, /*min_count=*/3);

bool g_ok = false;

}  // namespace

// -----------------------------------------------------------------------------

static void QuantizeFeatures() {
  const float scale = g_input->params.scale;
  const int   zp    = g_input->params.zero_point;
  int8_t*     dst   = g_input->data.int8;

  for (int i = 0; i < kFeatureElements; i++) {
    int32_t v = (int32_t)lrintf(g_features[i] / scale) + zp;
    if (v < -128) v = -128;
    if (v > 127)  v = 127;
    dst[i] = (int8_t)v;
  }
}

static void DequantizeOutput() {
  const float scale = g_output->params.scale;
  const int   zp    = g_output->params.zero_point;
  const int8_t* src = g_output->data.int8;

  for (int i = 0; i < kCategoryCount; i++) {
    g_scores[i] = ((float)src[i] - (float)zp) * scale;
  }
}

#ifdef ENABLE_GOLDEN_TEST
static void RunGoldenTest() {
  Serial.println(F("\n--- golden vector test ---"));

  // Stage 1: do our features match the notebook's get_features()?
  FeatureProviderCompute(kGoldenAudio, g_features);

  float max_abs_err = 0.0f;
  for (int i = 0; i < kFeatureElements; i++) {
    const float err = fabsf(g_features[i] - kGoldenFeatures[i]);
    if (err > max_abs_err) max_abs_err = err;
  }
  // Dynamic range is pinned to kLogTopRange by the clamp, so that is the
  // natural denominator. Expect ~3e-4 %; anything above 0.1% is a real bug.
  const float rel = max_abs_err / kLogTopRange;

  Serial.printf("  Feature max error  : %.3e  (%.5f%% of range)\n",
                max_abs_err, rel * 100.0f);
  if (rel < 1e-3f) {
    Serial.println(F("  Features ............ PASS"));
  } else {
    Serial.println(F("  Features ............ FAIL"));
    Serial.println(F("  -> check the Hann window is PERIODIC (divide by N),"));
    Serial.println(F("     that filterbank.h was regenerated alongside"));
    Serial.println(F("     model.h, and that kFFTSize matches the notebook."));
  }

  // Stage 2: does our interpreter match the desktop interpreter?
  QuantizeFeatures();
  if (g_interp->Invoke() != kTfLiteOk) {
    Serial.println(F("  Invoke .............. FAIL"));
    return;
  }

  int max_out_diff = 0;
  for (int i = 0; i < kCategoryCount; i++) {
    const int d = abs((int)g_output->data.int8[i] - (int)kGoldenOutput[i]);
    if (d > max_out_diff) max_out_diff = d;
  }

  DequantizeOutput();
  int best = 0;
  for (int i = 1; i < kCategoryCount; i++) {
    if (g_scores[i] > g_scores[best]) best = i;
  }

  Serial.printf("  Output max int8 diff: %d\n", max_out_diff);
  Serial.printf("  Predicted           : %s (%.3f)\n",
                kCategoryLabels[best], g_scores[best]);
  Serial.printf("  Expected            : %s\n",
                kCategoryLabels[kGoldenLabelIndex]);
  Serial.println(best == kGoldenLabelIndex ? F("  Inference ........... PASS")
                                           : F("  Inference ........... FAIL"));
  Serial.println(F("--------------------------\n"));
}
#endif

// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println(F("\nMIDI note classifier -- ESP32"));

  if (!FeatureProviderInit()) {
    Serial.println(F("FATAL: feature geometry mismatch. Check kFFTSize, "
                     "kNumFFTBins and kNumLogBins in model_params.h against "
                     "filterbank.h -- regenerate BOTH from the notebook."));
    return;
  }

  g_model_ptr = tflite::GetModel(g_model);
  if (g_model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("FATAL: schema %lu, expected %d. Regenerate model.h with a "
                  "TF version matching the tflm_esp32 library.\n",
                  (unsigned long)g_model_ptr->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Exactly the ops in the distilled student: Conv2D -> MaxPool2D ->
  // Flatten(Reshape) -> Dense -> Dense -> Softmax.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter interpreter(
      g_model_ptr, resolver, g_tensor_arena, kTensorArenaSize);
  g_interp = &interpreter;

  if (g_interp->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("FATAL: AllocateTensors failed. Raise kTensorArenaSize, "
                     "or check you exported WITHOUT "
                     "tf.lite.Optimize.EXPERIMENTAL_SPARSITY -- TFLM cannot "
                     "decode the sparse tensor format."));
    return;
  }

  g_input  = g_interp->input(0);
  g_output = g_interp->output(0);

  Serial.printf("Arena used   : %u / %u bytes\n",
                (unsigned)g_interp->arena_used_bytes(),
                (unsigned)kTensorArenaSize);
  Serial.printf("Input        : %d log bins int8, scale %.8f zp %d\n",
                g_input->dims->data[1],
                g_input->params.scale, g_input->params.zero_point);
  Serial.printf("Output       : %d classes, scale %.8f zp %d\n",
                g_output->dims->data[1],
                g_output->params.scale, g_output->params.zero_point);

  if (g_input->dims->data[1] != kNumLogBins) {
    Serial.println(F("FATAL: model input shape disagrees with model_params.h"));
    return;
  }

#ifdef ENABLE_GOLDEN_TEST
  RunGoldenTest();
#endif

  if (!AudioProviderInit()) {
    Serial.println(F("FATAL: I2S init failed. Check wiring and pin defines."));
    return;
  }

  g_ok = true;
  Serial.println(F("Listening.\n"));
}

void loop() {
  if (!g_ok) {
    delay(1000);
    return;
  }

  if (!AudioProviderReadWindow(g_audio, kHopSamples)) return;

  const float rms = AudioProviderLastRms();
  if (rms < kRmsGate) return;

  const uint32_t t0 = micros();
  FeatureProviderCompute(g_audio, g_features);
  const uint32_t t1 = micros();

  QuantizeFeatures();
  if (g_interp->Invoke() != kTfLiteOk) return;
  DequantizeOutput();
  const uint32_t t2 = micros();

  const char* label = nullptr;
  float       score = 0.0f;
  bool        is_new = false;
  g_recognizer.Update(g_scores, (int32_t)millis(), &label, &score, &is_new);

  if (is_new) {
    // Same output format as Lab 5's command responder, for continuity.
    Serial.printf("Heard %s (%d) @%lums\n",
                  label, (int)(score * 255.0f), (unsigned long)millis());
  }

  // Uncomment for diagnostics. Features are gain-invariant, so RMS only needs
  // to clear kRmsGate -- it no longer has to hit a target band. Check that
  // feat+infer stays well under the 32 ms hop.
  // Serial.printf("rms %.4f | feat %luus | infer %luus | top %s %.2f\n",
  //               rms, (unsigned long)(t1 - t0), (unsigned long)(t2 - t1),
  //               label, score);
  (void)t0; (void)t1; (void)t2;
}
