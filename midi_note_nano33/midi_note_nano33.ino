// =============================================================================
//  MIDI note classifier on Arduino Nano 33 BLE Sense
//  EE 446 final project -- second deployment target
// =============================================================================
//
//  Pipeline:  PDM mic -> 128 ms window -> 2048-point FFT
//             -> log-frequency filterbank (217 bins, 3/semitone)
//             -> log + max-relative floor + mean subtraction
//             -> int8 quantize -> TFLM interpreter -> softmax
//             -> posterior smoothing -> Serial + BLE MIDI
//
//  BOARD: must be the "Nano 33 BLE Sense" (rev 1 or 2). The plain
//  "Nano 33 BLE" has no microphone.
//
//  ---- Libraries -------------------------------------------------------------
//  1. Boards Manager -> "Arduino Mbed OS Nano Boards"
//  2. Library Manager -> "Chirale_TensorFlowLite"
//       The Arduino_TensorFlowLite library used in Lab 5 was removed from the
//       Library Manager and is unmaintained. Chirale_TensorFlowLite is
//       regenerated from current TFLite Micro and tested on this board. If you
//       already have the Lab 5 library installed it will also work -- this
//       sketch detects which one is present.
//  3. Library Manager -> "ArduinoBLE"            (for BLE MIDI)
//  4. Library Manager -> "Arduino_CMSIS-DSP"     (optional, for speed --
//                                                 see feature_provider.cpp)
//
//  ---- Headers ---------------------------------------------------------------
//  Copy model.h, model_params.h, filterbank.h and golden_test.h from the
//  training notebook. They are IDENTICAL to the ESP32 build -- the model and
//  front end are board-independent; only capture and output differ.
// =============================================================================

// ---------------------------------------------------------------------------
// TFLM LIBRARY: Chirale_TensorFlowLite. Install exactly one.
//
// *** DO NOT KEEP BOTH LIBRARIES INSTALLED. ***
//
// Chirale_TensorFlowLite and Arduino_TensorFlowLite both provide the same
// "tensorflow/lite/..." header paths used just below. With both present the
// Arduino builder has to guess which library satisfies those includes, and it
// can resolve headers from one while linking objects from the other -- an ODR
// violation with unpredictable behaviour and performance. If your compile
// output contains "Multiple libraries were found for ...", this is happening.
//
// Arduino_TensorFlowLite cannot be removed through Library Manager (it was
// delisted), so delete the folder by hand:
//     Windows   %USERPROFILE%\Documents\Arduino\libraries\Arduino_TensorFlowLite
//     macOS     ~/Documents/Arduino/libraries/Arduino_TensorFlowLite
//     Linux     ~/Arduino/libraries/Arduino_TensorFlowLite
//
// Why Chirale is the one to keep:
//   * maintained, in Library Manager, regenerated from current TFLite Micro.
//     The other is a 2021 2.4.0-ALPHA that Arduino delisted.
//   * declares architectures=mbed_nano, so it is built for this board.
//   * COMPILES FROM SOURCE. Arduino_TensorFlowLite sets precompiled=full and
//     links a prebuilt .a whose build flags you can neither inspect nor
//     change -- including whether CMSIS-NN got its Cortex-M4 DSP fast path.
//
// On kernels, verified by inspecting both source trees: BOTH ship the CMSIS-NN
// accelerated kernels (kernels/cmsis_nn/conv.cpp calling arm_convolve_wrapper_s8,
// plus fully_connected, pooling and softmax), and NEITHER ships a reference
// conv.cpp. So neither library is inherently slower. The advantage of Chirale is
// that it is compiled in your toolchain, where the flags are visible.
//
// This also corrects an earlier note in this file: the measured 116 ms
// (6.8 cycles/MAC) is NOT evidence of reference kernels, since reference
// kernels are not shipped at all. See the README for what to check instead.
// ---------------------------------------------------------------------------
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_provider.h"
#include "feature_provider.h"
#include "model.h"
#include "model_params.h"
#include "note_recognizer.h"

// Set to 0 to build without BLE (saves ~15 KB RAM, useful when debugging).
#define ENABLE_BLE_MIDI 1

// Set to 1 to run a timing benchmark at boot instead of listening. The numbers
// it prints are the on-device latency figures the report asks for.
#define ENABLE_BENCHMARK 1

// Comment out once the port is trusted.
#define ENABLE_GOLDEN_TEST
#ifdef ENABLE_GOLDEN_TEST
#include "golden_test.h"
#endif

#if ENABLE_BLE_MIDI
#include "ble_midi.h"
#endif

// -----------------------------------------------------------------------------
// Tuning
// -----------------------------------------------------------------------------

// 2048 samples = 128 ms, i.e. EQUAL TO THE WINDOW. The ESP32 uses 512.
//
// Two hard limits pin this value on the Nano:
//
//   Upper: the hop must not exceed the 128 ms window, or there are GAPS in
//          coverage -- stretches of audio never analysed, so a short note can
//          be missed entirely.
//   Lower: the hop must exceed (feature + inference) time, or the loop cannot
//          keep up and starts discarding audio.
//
// MEASURED on hardware: 17.5 ms features + 116 ms inference = 133.6 ms, which
// is ALREADY ABOVE the 128 ms upper limit. There is no hop that satisfies both
// until the pipeline gets faster. In order of effort:
//
//   1. Set USE_CMSIS_DSP 1 in feature_provider.cpp   (17.5 ms -> ~5 ms)
//   2. Retrain the smaller "Nano-friendly" student   (116 ms -> ~40 ms)
//
// Until then it still runs, just dropping backlog -- AudioProviderOverruns()
// counts what was lost. Detection latency is set by the 128 ms window either
// way, on both boards.
constexpr int kHopSamples = 2048;

constexpr float kRmsGate = 0.004f;

// Confidence needed before a note is sent over MIDI.
constexpr float kMidiThreshold = 0.60f;

// ---------------------------------------------------------------------------
// Smoothing, counted in FRAMES rather than milliseconds.
//
// This matters more than it looks. An earlier version specified the averaging
// window in wall-clock milliseconds, which fails silently on a slow board: if
// one inference takes longer than the window, fewer frames land inside it than
// the confirmation count requires, every Update() bails out, and the device
// reports NOTHING while looking perfectly healthy. Measured here: 133.6 ms per
// frame against a 160 ms window gave 2 frames where 3 were demanded.
//
// Counting frames is invariant to how fast the board happens to be, so the
// same numbers work on a 64 MHz M4 and a 240 MHz Xtensa.
// ---------------------------------------------------------------------------
constexpr int32_t kHopMs = (int32_t)((int32_t)kHopSamples * 1000 / kSampleRate);

constexpr int     kHistoryFrames   = 3;   // frames averaged together
constexpr int     kFramesToConfirm = 2;   // frames needed before reporting
constexpr int32_t kSuppressionMs   = kHopMs * 3;

constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

// ---------------------------------------------------------------------------
// Serial startup.
//
// The Nano 33 BLE does NOT reset when you open the Serial Monitor. Without
// waiting here, every boot message is printed before the monitor attaches and
// you see an empty window -- which looks identical to a crashed board.
//
// We wait up to kSerialWaitMs, then continue anyway so the device still works
// standalone during the demo with no PC attached.
// ---------------------------------------------------------------------------
constexpr uint32_t kSerialWaitMs = 15000;

// The RGB LED is active LOW on this board. It reports progress even when no
// Serial Monitor is attached, so "board is dead" and "board is running but you
// cannot see it" are distinguishable.
#define LED_OFF(p) digitalWrite((p), HIGH)
#define LED_ON(p)  digitalWrite((p), LOW)

// Blinks red `code` times, forever. Each failure below has its own count so the
// stage is identifiable without Serial.
static void FailStop(int code) {
  while (true) {
    for (int i = 0; i < code; i++) {
      LED_ON(LEDR);  delay(200);
      LED_OFF(LEDR); delay(200);
    }
    delay(1200);
  }
}

// -----------------------------------------------------------------------------

namespace {

const tflite::Model*      g_model_ptr = nullptr;
tflite::MicroInterpreter* g_interp    = nullptr;
TfLiteTensor*             g_input     = nullptr;
TfLiteTensor*             g_output    = nullptr;

float g_audio[kWindowSamples];
float g_features[kFeatureElements];
float g_scores[kCategoryCount];

NoteRecognizer g_recognizer(kHistoryFrames, /*detection_thresh=*/0.55f,
                            kSuppressionMs, kFramesToConfirm);

bool g_ok = false;
int  g_sounding = -1;          // MIDI note currently held on, or -1

const char* kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F",
                              "F#", "G", "G#", "A", "A#", "B"};

int LabelToMidi(const char* label) {
  if (!label) return -1;
  if (label[0] < '0' || label[0] > '9') return -1;   // "silence"
  return atoi(label);
}

void NoteName(int midi, char* buf, size_t n) {
  if (midi < 0) { snprintf(buf, n, "silence"); return; }
  snprintf(buf, n, "%s%d", kNoteNames[midi % 12], (midi / 12) - 1);
}

void QuantizeFeatures() {
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

void DequantizeOutput() {
  const float scale = g_output->params.scale;
  const int   zp    = g_output->params.zero_point;
  const int8_t* src = g_output->data.int8;
  for (int i = 0; i < kCategoryCount; i++) {
    g_scores[i] = ((float)src[i] - (float)zp) * scale;
  }
}

}  // namespace

#ifdef ENABLE_GOLDEN_TEST
static void RunGoldenTest() {
  Serial.println(F("\n--- golden vector test ---"));

  FeatureProviderCompute(kGoldenAudio, g_features);
  float max_err = 0.0f;
  for (int i = 0; i < kFeatureElements; i++) {
    const float e = fabsf(g_features[i] - kGoldenFeatures[i]);
    if (e > max_err) max_err = e;
  }
  const float rel = max_err / kLogTopRange;
  Serial.print(F("  Feature max error  : ")); Serial.print(max_err, 8);
  Serial.print(F("  (")); Serial.print(rel * 100.0f, 5); Serial.println(F("% of range)"));
  if (rel < 1e-3f) {
    Serial.println(F("  Features ............ PASS"));
  } else {
    Serial.println(F("  Features ............ FAIL"));
    Serial.println(F("  -> if USE_CMSIS_DSP is 1 in feature_provider.cpp, set it"));
    Serial.println(F("     back to 0 and re-test: the CMSIS real-FFT output"));
    Serial.println(F("     packing is the most likely culprit."));
  }

  QuantizeFeatures();
  if (g_interp->Invoke() != kTfLiteOk) {
    Serial.println(F("  Invoke .............. FAIL"));
    return;
  }
  DequantizeOutput();
  int best = 0;
  for (int i = 1; i < kCategoryCount; i++) {
    if (g_scores[i] > g_scores[best]) best = i;
  }
  Serial.print(F("  Predicted           : ")); Serial.println(kCategoryLabels[best]);
  Serial.print(F("  Expected            : ")); Serial.println(kCategoryLabels[kGoldenLabelIndex]);
  Serial.println(best == kGoldenLabelIndex ? F("  Inference ........... PASS")
                                           : F("  Inference ........... FAIL"));
  Serial.println(F("--------------------------\n"));
}
#endif

// Always run (cheaply) even when the printout is disabled, because the
// recognizer is configured from the result.
static float MeasureFrameMs(int N, bool verbose) {
  if (verbose) Serial.println(F("--- latency benchmark (report these numbers) ---"));

  uint32_t f_tot = 0, f_max = 0, i_tot = 0, i_max = 0;
  for (int r = 0; r < N; r++) {
#ifdef ENABLE_GOLDEN_TEST
    const float* src = kGoldenAudio;
#else
    const float* src = g_audio;
#endif
    uint32_t t0 = micros();
    FeatureProviderCompute(src, g_features);
    uint32_t t1 = micros();
    QuantizeFeatures();
    g_interp->Invoke();
    uint32_t t2 = micros();

    const uint32_t f = t1 - t0, in = t2 - t1;
    f_tot += f; i_tot += in;
    if (f > f_max) f_max = f;
    if (in > i_max) i_max = in;
  }

  const float f_avg = f_tot / (float)N / 1000.0f;
  const float i_avg = i_tot / (float)N / 1000.0f;

  if (!verbose) return f_avg + i_avg;

  Serial.print(F("  feature extraction : ")); Serial.print(f_avg, 2);
  Serial.print(F(" ms avg, ")); Serial.print(f_max / 1000.0f, 2); Serial.println(F(" ms max"));
  Serial.print(F("  inference          : ")); Serial.print(i_avg, 2);
  Serial.print(F(" ms avg, ")); Serial.print(i_max / 1000.0f, 2); Serial.println(F(" ms max"));
  Serial.print(F("  TOTAL per frame    : ")); Serial.print(f_avg + i_avg, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  hop budget         : "));
  Serial.print(kHopSamples * 1000.0f / kSampleRate, 1); Serial.println(F(" ms"));

  const float total = f_avg + i_avg;
  const float hop_ms = kHopSamples * 1000.0f / kSampleRate;
  const float win_ms = kWindowSamples * 1000.0f / kSampleRate;

  Serial.print(F("  window (max usable hop): ")); Serial.print(win_ms, 1);
  Serial.println(F(" ms"));

  if (total > win_ms) {
    Serial.println(F("  PROBLEM: processing exceeds the window length, so there"));
    Serial.println(F("           is NO hop that both keeps up and covers all"));
    Serial.println(F("           audio. It will run, but drop frames."));
    Serial.println(F("           Fix 1: USE_CMSIS_DSP 1 in feature_provider.cpp"));
    Serial.println(F("           Fix 2: retrain the Nano-friendly student"));
  } else if (total > hop_ms * 0.8f) {
    Serial.print(F("  WARNING: uses >80% of the hop. Raise kHopSamples to >= "));
    Serial.println((int)(total * 1.25f * kSampleRate / 1000.0f));
  } else {
    Serial.println(F("  OK: comfortable headroom."));
  }

  // Frame-counted smoothing cannot starve, so this is informational rather
  // than a failure check: it tells you how responsive the device will feel.
  Serial.print(F("  smoothing: ")); Serial.print(kFramesToConfirm);
  Serial.print(F(" of ")); Serial.print(kHistoryFrames);
  Serial.print(F(" frames = ~")); Serial.print((int)(total * kHistoryFrames));
  Serial.println(F(" ms of history"));
  Serial.print(F("  note onset -> first report: ~"));
  Serial.print((int)((float)kWindowSamples * 1000.0f / kSampleRate
                     + total * kFramesToConfirm));
  Serial.println(F(" ms"));
  Serial.println(F("-----------------------------------------------\n"));
  return f_avg + i_avg;
}

void setup() {
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  LED_OFF(LEDR); LED_OFF(LEDG); LED_OFF(LEDB);

  Serial.begin(115200);

  // Blue blink = waiting for you to open the Serial Monitor.
  const uint32_t t_wait = millis();
  while (!Serial && (millis() - t_wait) < kSerialWaitMs) {
    LED_ON(LEDB);  delay(100);
    LED_OFF(LEDB); delay(400);
  }
  LED_OFF(LEDB);

  Serial.println();
  Serial.println(F("MIDI note classifier -- Nano 33 BLE Sense"));
  Serial.print(F("serial attached after ")); Serial.print(millis() - t_wait);
  Serial.println(F(" ms"));

  if (!FeatureProviderInit()) {
    Serial.println(F("FATAL: feature geometry mismatch. Regenerate "
                     "model_params.h and filterbank.h together."));
    FailStop(2);
  }

  g_model_ptr = tflite::GetModel(g_model);
  if (g_model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("FATAL: model schema mismatch. Regenerate model.h with a "
                     "TF version matching the TFLM library."));
    FailStop(3);
  }

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
    Serial.println(F("FATAL: AllocateTensors failed -- raise kTensorArenaSize, "
                     "or check the model was exported WITHOUT "
                     "EXPERIMENTAL_SPARSITY."));
    FailStop(4);
  }

  g_input  = g_interp->input(0);
  g_output = g_interp->output(0);

  Serial.print(F("Arena used : ")); Serial.print(g_interp->arena_used_bytes());
  Serial.print(F(" / ")); Serial.print(kTensorArenaSize);
  Serial.println(F(" bytes   <-- report this as RAM"));
  Serial.print(F("Classes    : ")); Serial.println(kCategoryCount);

  if (g_input->dims->data[1] != kNumLogBins) {
    Serial.println(F("FATAL: model input shape disagrees with model_params.h"));
    FailStop(5);
  }

#ifdef ENABLE_GOLDEN_TEST
  RunGoldenTest();
#endif

  // Measure the real frame time, then size the smoothing window to match.
  //
  // This is not cosmetic. The default 160 ms history with min_count=3 needs
  // three inferences inside 160 ms; at 134 ms per frame only two ever land, so
  // Update() bails out early every single time and no note is EVER reported.
  // The board looks like it is working and prints nothing.
  const float frame_ms = MeasureFrameMs(20, ENABLE_BENCHMARK);

  // The history is counted in FRAMES, so it needs no measurement -- that is the
  // whole point. Only the wall-clock parameters are derived from frame_ms:
  //   suppression -- must exceed one frame, or a held note retriggers every frame
  //   max age     -- must comfortably span the history, or frames expire before
  //                  they can be averaged
  int32_t supp = (int32_t)(frame_ms * 2.0f);
  if (supp < 250) supp = 250;

  int32_t max_age = (int32_t)(frame_ms * kHistoryFrames * 3.0f);
  if (max_age < 2000) max_age = 2000;

  g_recognizer.Configure(kHistoryFrames, /*detection_thresh=*/0.50f, supp,
                         kFramesToConfirm, max_age);

  Serial.print(F("smoothing: ")); Serial.print(kFramesToConfirm);
  Serial.print(F(" of ")); Serial.print(kHistoryFrames);
  Serial.print(F(" frames = ")); Serial.print((int)(frame_ms * kHistoryFrames));
  Serial.print(F(" ms; suppression ")); Serial.print(supp);
  Serial.print(F(" ms  (frame time "));
  Serial.print(frame_ms, 1); Serial.println(F(" ms)"));
  Serial.print(F("note onset -> first report: ~"));
  Serial.print((int)((float)kWindowSamples * 1000.0f / kSampleRate
                     + frame_ms * kFramesToConfirm));
  Serial.println(F(" ms"));
  Serial.print(F("update rate: ")); Serial.print(1000.0f / frame_ms, 1);
  Serial.println(F(" Hz\n"));

  if (!AudioProviderInit()) {
    Serial.println(F("FATAL: PDM init failed. Is this a Nano 33 BLE *Sense*? "
                     "The non-Sense board has no microphone."));
    FailStop(6);
  }

#if ENABLE_BLE_MIDI
  if (!BleMidiInit()) {
    Serial.println(F("WARNING: BLE init failed; continuing with Serial only."));
  } else {
    Serial.print(F("BLE MIDI advertising as \""));
    Serial.print(kDeviceName);
    Serial.println(F("\" -- pair from your host before the demo."));
  }
#endif

  g_ok = true;
  LED_ON(LEDG);
  Serial.println(F("Listening. A heartbeat prints every 5 s so you can tell"));
  Serial.println(F("the difference between 'no notes detected' and 'hung'.\n"));
}

void loop() {
  if (!g_ok) { delay(1000); return; }

  // Heartbeat: without this, a silent room and a hung loop look identical.
  static uint32_t last_beat = 0;
  if (millis() - last_beat > 5000) {
    last_beat = millis();
    Serial.print(F("[alive] rms ")); Serial.print(AudioProviderLastRms(), 5);
    Serial.print(F("  gate ")); Serial.print(kRmsGate, 5);
    // Best current guess even when it has not cleared the reporting threshold.
    // If this tracks the note you are playing but nothing prints, the
    // confidence threshold is too high rather than the model being wrong.
    int best = 0;
    for (int i = 1; i < kCategoryCount; i++) {
      if (g_scores[i] > g_scores[best]) best = i;
    }
    Serial.print(F("  top ")); Serial.print(kCategoryLabels[best]);
    Serial.print(F(" ")); Serial.print(g_scores[best], 2);
    Serial.print(F("  overruns ")); Serial.print(AudioProviderOverruns());
    Serial.print(F("  frames_avg ")); Serial.print(g_recognizer.LastUsedFrames());
#if ENABLE_BLE_MIDI
    Serial.print(F("  ble ")); Serial.print(BleMidiConnected() ? F("connected")
                                                              : F("advertising"));
#endif
    Serial.println();
  }

#if ENABLE_BLE_MIDI
  BleMidiPoll();
#endif

  if (!AudioProviderReadWindow(g_audio, kHopSamples)) return;

  if (AudioProviderLastRms() < kRmsGate) {
    // Silence: release any held note so the synth does not drone.
    if (g_sounding >= 0) {
#if ENABLE_BLE_MIDI
      BleMidiNoteOff(g_sounding);
#endif
      g_sounding = -1;
    }
    return;
  }

  FeatureProviderCompute(g_audio, g_features);
  QuantizeFeatures();
  if (g_interp->Invoke() != kTfLiteOk) return;
  DequantizeOutput();

  // Report the true frame period once, so the figure on your slide comes from
  // the running system rather than from the idle benchmark.
  static uint32_t prev_frame_ms = 0;
  static bool     reported_period = false;
  const uint32_t now_ms = millis();
  if (prev_frame_ms && !reported_period) {
    Serial.print(F("[timing] live frame period ")); Serial.print(now_ms - prev_frame_ms);
    Serial.print(F(" ms -> ")); Serial.print(1000.0f / (now_ms - prev_frame_ms), 1);
    Serial.println(F(" inferences/s"));
    reported_period = true;
  }
  prev_frame_ms = now_ms;

  const char* label = nullptr;
  float score = 0.0f;
  bool  is_new = false;
  g_recognizer.Update(g_scores, (int32_t)millis(), &label, &score, &is_new);

  if (is_new) {
    char name[8];
    const int midi = LabelToMidi(label);
    NoteName(midi, name, sizeof(name));
    Serial.print(F("Heard ")); Serial.print(name);
    Serial.print(F(" (")); Serial.print(label);
    Serial.print(F(")  conf ")); Serial.print(score, 2);
    Serial.print(F("  @")); Serial.print(millis()); Serial.println(F("ms"));
  }

  // ---- MIDI note tracking ---------------------------------------------------
  // Driven by the smoothed label rather than by is_new, so that note-OFF is sent
  // when the pitch changes or drops below confidence -- otherwise a held note
  // would sustain forever on the synth.
  if (score >= kMidiThreshold) {
    const int midi = LabelToMidi(label);
    if (midi != g_sounding) {
      if (g_sounding >= 0) {
#if ENABLE_BLE_MIDI
        BleMidiNoteOff(g_sounding);
#endif
      }
      if (midi >= 0) {
#if ENABLE_BLE_MIDI
        BleMidiNoteOn(midi, (int)(score * 127.0f));
#endif
      }
      g_sounding = midi;
    }
  }

  // Uncomment to watch for the loop falling behind the microphone.
  // static uint32_t last = 0;
  // if (AudioProviderOverruns() != last) {
  //   last = AudioProviderOverruns();
  //   Serial.print(F("overruns: ")); Serial.println(last);
  // }
}
