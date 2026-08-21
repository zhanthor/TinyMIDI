// =============================================================================
//  MIDI note classifier on ESP32
//  EE 446 final project -- deployment via the Lab 5 workflow
// =============================================================================
//
//  Pipeline:  I2S mic -> 128 ms window -> 2048-point FFT
//             -> log-frequency filterbank (29x65 bins)
//             -> log + max-relative floor + mean subtraction
//             -> int8 quantize -> TFLM interpreter -> softmax
//             -> posterior smoothing -> serial/MIDI
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
// #include "note_recognizer.h"

// -----------------------------------------------------------------------------
// Tuning
// -----------------------------------------------------------------------------

// 128 ms analysis window advances per inference. 512 samples = 32 ms.
// However Inference takes about 300ms.
//
// The hop sets the UPDATE rate, not the detection latency: a newly
// struck note is not fully inside the window until 128 ms have passed.
constexpr int kHopSamples = 512;

// Skip inference entirely below this RMS. The model has a `silence` class
// but this it keeps the CPU idle in a quiet room and
// stops the recognizer's history filling with noise-driven garbage.
constexpr float kRmsGate = 0.1f;

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


//Audio input from INMP441 - see audio_provider.ccp/.h
//  SD   GPIO 32
//  WS   GPIO 25
//  SCK  GPIO 32
//  L/R  GND 

float* g_audio = new float[kWindowSamples];
float* g_features = new float[kFeatureElements];
float* g_scores = new float[kCategoryCount];

// NoteRecognizer g_recognizer;

bool g_ok = false;

} 

// -----------------------------------------------------------------------------

#define MIDI_BAUD 31250

HardwareSerial MidiSerial(2);

constexpr int kMidiTxPin = 17;   // ESP32 UART2 TX2

int g_current_note = -1;
int g_candidate_note = -1;
int g_candidate_count = 0;

constexpr float kMidiThreshold = 0.30f;
constexpr int kStableFrames = 2;
int g_off_count = 0;
int g_on_count = 0;
constexpr int kOffStableFrames = 2;
constexpr float kMidiThresholdOn = 0.60f;
constexpr float kMidiThresholdOff = 0.30f;

void MidiNoteOn(int note, int velocity) {
  // Serial.println(">>> MidiNoteOn called");
  // Serial.printf("UART TX bytes: 0x%02X 0x%02X 0x%02X\n",
  //             0x90,
  //             (unsigned)(note & 0x7F),
  //             (unsigned)(velocity & 0x7F));
  MidiSerial.write(0x90);  // channel 1
  MidiSerial.write(note & 0x7F); // note(0-127)
  MidiSerial.write(velocity & 0x7F); 
}

void MidiNoteOff(int note) {
  // Serial.printf("UART TX bytes: 0x90 0x%02X 0x%02X\n",
  //               (unsigned)(note & 0x7F));
  MidiSerial.write(0x80);  // channel 1
  MidiSerial.write(note & 0x7F);
  MidiSerial.write(0);
}

//----------------------------------------------------

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

// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1500);

  MidiSerial.begin(MIDI_BAUD, SERIAL_8N1, -1, kMidiTxPin);

  g_audio = (float*)malloc(sizeof(float) * kWindowSamples);
  g_features = (float*)malloc(sizeof(float) * kFeatureElements);
  g_scores = (float*)malloc(sizeof(float) * kCategoryCount);

  if (!g_audio || !g_features || !g_scores) {
    Serial.println("FATAL: malloc failed");
    while (true) {
      delay(1000);
    }
  }

  if (g_audio == nullptr ||
      g_features == nullptr ||
      g_scores == nullptr) {
    Serial.println(F("FATAL: float buffer allocation failed"));
    return;
  }

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

  if (g_input->dims->data[1] != 29 ||
    g_input->dims->data[2] != 65 ||
    g_input->dims->data[3] != 1) {
  Serial.println("FATAL: model input shape disagrees with model");
  while (true) delay(1000);
}

  if (!AudioProviderInit()) {
    Serial.println(F("FATAL: I2S init failed. Check wiring and pin defines."));
    return;
  }

  g_ok = true;
  Serial.println(F("Listening.\n"));
}

void loop() {
  if (!g_ok) {
    return;
  }

  if (!AudioProviderReadWindow(g_audio, kHopSamples)) return;

  const float rms = AudioProviderLastRms();
  if (rms < kRmsGate) return;

  // ---------------------------------------------------------------------------
  // 1) Features, 2) quantize, 3) inference, 4) dequantize
  // ---------------------------------------------------------------------------
  const uint32_t t0 = micros();
  FeatureProviderCompute(g_audio, g_features);
  const uint32_t t1 = micros();

  QuantizeFeatures();

  if (g_interp->Invoke() != kTfLiteOk) {
    Serial.println("INFERENCE FAILED");
    return;
  }

  DequantizeOutput();
  const uint32_t t2 = micros();

  
  // Pick the highest-probability class
  int best = 0;
  float best_score = g_scores[0];

  for (int i = 1; i < kCategoryCount; i++) {
    if (g_scores[i] > best_score) {
      best_score = g_scores[i];
      best = i;
    }
  }

  Serial.printf(
      "RMS %.4f | RAW: class %d = %s | score %.3f\n",
      rms,
      best,
      kCategoryLabels[best],
      best_score
  );

  // ---------------------------------------------------------------------------
  // Convert classifier class -> MIDI note.
  // Classes 0..26 correspond to MIDI notes 45..71.
  // Class 27 is silence.
  // ---------------------------------------------------------------------------

  if (best == kSilenceIndex || best_score < kMidiThreshold) {

    // Not confident enough to call a note.
    g_candidate_note = -1;
    g_candidate_count = 0;
    g_off_count++;

    if (g_off_count >= kOffStableFrames && g_current_note >= 0) {
      MidiNoteOff(g_current_note);

      Serial.printf(
        "MIDI OFF: %d\n",
        g_current_note
      );

      g_current_note = -1;
      g_off_count = 0;
    }

  } else {
    const int candidate_note = 45 + best;

    // Same candidate as last inference.
    if (candidate_note == g_candidate_note) {
      g_candidate_count++;
    } else {
      // New candidate.
      g_candidate_note = candidate_note;
      g_candidate_count = 1;
    }

    // Require a few consecutive classifications before changing note.
    if (g_candidate_count >= kStableFrames &&
        candidate_note != g_current_note) {

      // Turn off previous note.
      if (g_current_note >= 0) {
        MidiNoteOff(g_current_note);

        Serial.printf(
          "MIDI OFF: %d\n",
          g_current_note
        );
      }

      // Convert classifier confidence to MIDI velocity.
      int velocity = (int)(best_score * 127.0f);

      if (velocity < 20) velocity = 20;
      if (velocity > 127) velocity = 127;

      MidiNoteOn(candidate_note, velocity);

      Serial.printf(
        "MIDI ON: %d  confidence %.3f  velocity %d\n",
        candidate_note,
        best_score,
        velocity
      );

      g_current_note = candidate_note;
    }
  }

  (void)t0;
  (void)t1;
  (void)t2;
  Serial.print("Feat extraction latency:");
  Serial.println(t1-t0);
  Serial.print("Inference latency:");
  Serial.println(t2-t1);
  Serial.print("Total end-to-end latency:");
  Serial.println(t2-t0);
}
