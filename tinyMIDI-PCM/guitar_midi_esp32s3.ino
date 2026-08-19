/*
 * TinyML Guitar-to-MIDI -- ESP32-S3 USB-MIDI firmware (raw-audio, dual-head)
 * ============================================================================
 *
 * Reads a mono audio stream (INMP441 I2S mic, or a Pmod I2S2 line-in ADC), runs a sliding
 * window through an on-device TFLite Micro model with two heads -- note pitch and onset --
 * and emits USB-MIDI Note On/Off messages in real time. Companion training pipeline:
 * train_guitar_midi.ipynb.
 *
 * ---------------------------------------------------------------------
 * HOW TIMING WORKS (read this before changing any of the tuning constants below)
 * ---------------------------------------------------------------------
 * Every gate in the note state machine is evaluated against micros() timestamps carried
 * alongside the audio (AudioWindowMsg::windowEndMicros -> InferenceResultMsg::windowEndMicros),
 * not against a frame count. That is deliberate: a frame-count gate is only correct at the
 * inference speed it was tuned for, and this model's inference time depends on MODEL_WIDTH /
 * STEM_STRIDE in the training notebook, which changes across retrains. Timestamp-driven gates
 * mean what their names say (in milliseconds) at any inference speed.
 *
 * The firmware measures its own Invoke() time continuously and prints it in a periodic
 * [HEALTH] line, compared against TARGET_INFERENCE_PERIOD_MS. WATCH THAT LINE FIRST when
 * diagnosing any timing symptom (dropped notes, late notes, inconsistent latency): if Invoke()
 * doesn't fit comfortably inside the target period, the onset head is being sampled past its
 * positive band and no amount of threshold tuning will fix it -- the model needs to be made
 * cheaper in the notebook (MODEL_WIDTH / STEM_STRIDE), not the firmware retuned.
 *
 * Design points that follow from this:
 *  - NOTE ON FIRES ON THE ONSET ITSELF, not after a stability-count gate. The note state
 *    machine is IDLE <-> SUSTAIN; there is no separate ATTACK state. Firing on the onset keeps
 *    the pluck-to-Note-On delay constant across notes, which matters more than making that
 *    delay small: a DAW can compensate a fixed offset, but not a delay that varies note to
 *    note.
 *  - RETRIGGER_LOCKOUT_MS exists because one pluck produces several consecutive onset-positive
 *    windows (ONSET_WINDOW_MS wide). Without a lockout, a fast inference loop would emit one
 *    Note On per positive window instead of one per pluck.
 *  - SAME-PITCH REPEATS RETRIGGER. Retrigger keys on the onset firing, not on the pitch
 *    changing -- repeated notes at the same pitch are extremely common in real playing.
 *  - RELEASE HAS TWO SAFETY NETS BEYOND THE MODEL'S OWN SILENCE CLASSIFICATION: an RMS-based
 *    release (level under RELEASE_RMS_DBFS for RELEASE_HOLD_MS) and a hard
 *    MAX_NOTE_DURATION_MS ceiling. A stuck Note On is the worst failure mode a MIDI instrument
 *    has -- audible indefinitely, and needs a panic message to clear -- so release does not
 *    depend solely on the model being right.
 *  - THE MODE (VOTING) FILTER IS TIME-WINDOWED (MODE_FILTER_WINDOW_MS), not count-windowed, so
 *    its smoothing window means the same number of milliseconds regardless of inference speed.
 *    It is also RESET AND RESEEDED on every accepted onset: the onset decision and the Note On
 *    pitch always use the current frame's raw classification (the only evidence describing the
 *    note that just started), while the release decision uses the smoothed vote (informative
 *    only because every vote in it belongs to the note currently sounding, thanks to the
 *    reset). Judging a new note by pre-onset history fails in both directions -- a note after a
 *    rest gets judged against silence votes and is rejected outright, and a note after another
 *    note gets judged against the old pitch -- so the two decisions must not share evidence.
 *
 * ---------------------------------------------------------------------
 * ACCURACY / SIGNAL PATH DESIGN POINTS
 * ---------------------------------------------------------------------
 *  - quantize_and_load_input() normalizes unconditionally: divide by max(peak,
 *    MODEL_NORM_PEAK_FLOOR), matching normalize_window_np()/normalize_window_tf() in the
 *    training notebook exactly. This must stay in parity with the notebook -- see the
 *    notebook's "normalization parity check" cell.
 *  - WINDOW_SAMPLES, HOP_SAMPLES, SAMPLE_RATE, the normalization floor, the onset band width,
 *    and the onset decision threshold are NOT hardcoded here -- they come from
 *    class_labels.h, generated fresh by every run of the training notebook, so this file
 *    cannot silently drift out of sync with the model it's loading.
 *  - WINDOW_SAMPLES is 2048 (128ms): gives ~7.8 Hz of frequency resolution, which matters most
 *    for low-string semitone discrimination (E2/F2 are 4.9 Hz apart).
 *  - Model input tensor is int8 (T, 1, 1) matching the notebook's Conv2D/DepthwiseConv2D
 *    trunk (needed so quantization-aware training works at all -- tfmot has no
 *    DepthwiseConv1D support).
 *  - SELECTABLE AUDIO INPUT: onboard INMP441 I2S microphone, or a Digilent Pmod I2S2 (CS5343
 *    ADC) wired to a 3.5mm line-in jack, picked at compile time via AUDIO_SOURCE below. Both
 *    peripherals can be wired simultaneously on non-overlapping pins.
 *  - PMOD I2S2 WIRING: the Pmod's 12-pin connector is two INDEPENDENT I2S interfaces, not one
 *    signal set duplicated for daisy-chaining. Top row (1-6) is the D/A side driving line-OUT;
 *    bottom row (7-12) is the A/D side digitizing line-IN. This firmware wires the bottom row
 *    only -- see the wiring block below.
 *  - Built on esp-tflite-micro, which ships PREBUILT inside the esp32 Arduino core itself --
 *    zero extra library installs. If a third-party TensorFlowLite_ESP32 Arduino library is
 *    installed, remove it: its headers collide with the core-bundled ones if both are present
 *    on the include path.
 *
 * ---------------------------------------------------------------------
 * BOARD SETUP (Arduino IDE)
 * ---------------------------------------------------------------------
 *  - Board: an ESP32-S3 board with native USB (USB-OTG) pins wired out
 *    (most "ESP32-S3-DevKitC-1" boards qualify).
 *  - Tools > USB Mode: "USB-OTG (TinyUSB)"
 *  - Tools > USB CDC On Boot: "Disabled" or "Enabled" (either is fine;
 *    just don't rely on Serial for MIDI timing)
 *  - arduino-esp32 core >= 3.0 (ships USB.h / USBMIDI.h / the new
 *    driver/i2s_std.h I2S driver out of the box)
 *
 * ---------------------------------------------------------------------
 * LIBRARIES
 * ---------------------------------------------------------------------
 *  - None to install. This firmware targets esp-tflite-micro, which ships built into the
 *    esp32 Arduino core itself, along with esp-nn for accelerated INT8 conv on the S3's
 *    vector instructions. If the third-party TensorFlowLite_ESP32 Arduino library is
 *    installed, REMOVE it -- its headers collide with the core-bundled ones if both are
 *    present.
 *
 * ---------------------------------------------------------------------
 * FILES YOU NEED FROM THE TRAINING PIPELINE (same directory as this .ino)
 * ---------------------------------------------------------------------
 *  - model.h              exported by train_guitar_midi.ipynb
 *  - class_labels.h       exported alongside model.h -- CLASS_TO_MIDI[] plus every shared
 *                          constant. Regenerate BOTH together.
 *
 * ---------------------------------------------------------------------
 * WIRING -- pick ONE via AUDIO_SOURCE below (both may be wired at once)
 * ---------------------------------------------------------------------
 *
 * Option A: INMP441 I2S MEMS microphone
 *  Mic SCK  -> GPIO 4   (bit clock)
 *  Mic WS   -> GPIO 5   (word select / L-R clock)
 *  Mic SD   -> GPIO 6   (serial data out from mic)
 *  Mic L/R  -> GND      (mono, left channel)
 *  Mic VDD  -> 3V3, Mic GND -> GND
 *
 * Option B: Digilent Pmod I2S2 (CS5343 ADC for line-in), 3.5mm line-in jack (J2)
 *  Only the BOTTOM row (A/D side) is wired -- this firmware only reads audio in:
 *    Pmod pin 7  (A/D MCLK)  -> GPIO 15   -- REQUIRED, the CS5343 cannot derive its own clock
 *    Pmod pin 8  (A/D LRCK)  -> GPIO 17   (word select)
 *    Pmod pin 9  (A/D SCLK)  -> GPIO 16   (bit/serial clock)
 *    Pmod pin 10 (A/D SDOUT) -> GPIO 18   (ADC's data OUT -> our data IN)
 *    Pmod pin 11 (GND)       -> GND
 *    Pmod pin 12 (VCC)       -> 3V3
 *  The TOP row (pins 1-6) drives the line-OUT jack and is NOT wired. Its data pin, D/A SDIN,
 *  only ever carries data FROM the host INTO the DAC, so nothing the ADC captures could reach
 *  the ESP32 over that wiring. Confirm pin numbers against your board's silkscreen.
 */

#include "USB.h"
#include "USBMIDI.h"
#include "driver/i2s_std.h"

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"
#include "class_labels.h"

// ---------------------------------------------------------------------
// Sanity-check class_labels.h against this firmware revision.
// ---------------------------------------------------------------------
#ifndef MODEL_WINDOW_SAMPLES
#error "class_labels.h is missing MODEL_WINDOW_SAMPLES -- regenerate model.h/class_labels.h from the current train_guitar_midi.ipynb."
#endif
#ifndef MODEL_HOP_SAMPLES
#error "class_labels.h is missing MODEL_HOP_SAMPLES -- regenerate model.h/class_labels.h from the current train_guitar_midi.ipynb."
#endif
#ifndef MODEL_SAMPLE_RATE
#error "class_labels.h is missing MODEL_SAMPLE_RATE -- regenerate model.h/class_labels.h from the current train_guitar_midi.ipynb."
#endif
#ifndef MODEL_NORM_PEAK_FLOOR
#error "class_labels.h is missing MODEL_NORM_PEAK_FLOOR -- regenerate model.h/class_labels.h from the current train_guitar_midi.ipynb."
#endif
#ifndef MODEL_ONSET_THRESHOLD
#error "class_labels.h is missing MODEL_ONSET_THRESHOLD -- regenerate model.h/class_labels.h from the current train_guitar_midi.ipynb."
#endif

// MODEL_ONSET_WINDOW_MS is checked with #ifdef, NOT #error: older class_labels.h files
// (generated before the onset band width was exported) won't have it, and this firmware
// should still compile and run against them, falling back to the pre-export default (40ms)
// and saying so at boot rather than refusing to build.
#ifdef MODEL_ONSET_WINDOW_MS
static const int   ONSET_BAND_MS = MODEL_ONSET_WINDOW_MS;
static const bool  ONSET_BAND_FROM_MODEL = true;
#else
static const int   ONSET_BAND_MS = 40;
static const bool  ONSET_BAND_FROM_MODEL = false;
#endif

// NUM_NOTE_CLASSES is intentionally a `const int` in class_labels.h (not a #define), so it
// isn't visible to the preprocessor and can't be checked with #ifndef. No guard is needed:
// class_labels.h uses it two lines after declaring it to size CLASS_TO_MIDI[], so a genuinely
// missing NUM_NOTE_CLASSES already fails to compile there with a clear error.

// ---------------------------------------------------------------------
// Audio constants -- sourced from class_labels.h, NOT hardcoded here.
// ---------------------------------------------------------------------
static const int   SAMPLE_RATE      = MODEL_SAMPLE_RATE;
static const int   HOP_SAMPLES      = MODEL_HOP_SAMPLES;    // 10ms stride ("Buffer A")
static const int   WINDOW_SAMPLES   = MODEL_WINDOW_SAMPLES; // sliding window -> model input
static const float NORM_PEAK_FLOOR  = MODEL_NORM_PEAK_FLOOR;
static const float ONSET_PROB_THRESHOLD = MODEL_ONSET_THRESHOLD; // F1-swept on validation

static const int   HOP_PERIOD_US = (int)(1000000LL * HOP_SAMPLES / SAMPLE_RATE); // 10000us

// ---------------------------------------------------------------------
// Audio input source -- pick exactly ONE.
// ---------------------------------------------------------------------
#define AUDIO_SOURCE_INMP441_MIC 1
#define AUDIO_SOURCE_PMOD_I2S2   2

#ifndef AUDIO_SOURCE
#define AUDIO_SOURCE AUDIO_SOURCE_INMP441_MIC
#endif

// ---------------------------------------------------------------------
// I2S pins
// ---------------------------------------------------------------------
static const int MIC_BCLK_PIN = 4;
static const int MIC_WS_PIN   = 5;
static const int MIC_SD_PIN   = 6;

// The INMP441 transmits 24 significant bits, MSB-first, left-justified in each 32-bit I2S
// slot. Reading 32-bit slots and shifting right by 16 gives a 16-bit value roughly
// proportional to the mic's output level. If your captured signal is too quiet or clipped,
// retune this -- 11-14 is a common alternative range.
//
// GAIN TUNING: this sets the input path's effective sensitivity. If rmsToVelocity() (below)
// is consistently reporting levels near the bottom of its range during normal playing at a
// normal mic distance, lower this value (each step down doubles the captured level) rather
// than widening rmsToVelocity()'s dB range -- that just compresses dynamics instead of fixing
// the underlying gain.
static const int MIC_SHIFT_TO_INT16 = 16;

static const int PMOD_MCLK_PIN = 15; // Pmod pin 7  (A/D MCLK) -- REQUIRED
static const int PMOD_BCLK_PIN = 16; // Pmod pin 9  (A/D SCLK)
static const int PMOD_WS_PIN   = 17; // Pmod pin 8  (A/D LRCK)
static const int PMOD_SD_PIN   = 18; // Pmod pin 10 (A/D SDOUT)

// Line-level signals are typically much hotter than a MEMS mic, so this may need to be
// smaller than MIC_SHIFT_TO_INT16 to avoid clipping.
static const int PMOD_SHIFT_TO_INT16 = 16;

// The INMP441 is wired mono (L/R tied to GND); the Pmod I2S2's CS5343 is a true stereo ADC
// and always outputs both slots regardless of what's plugged into the jack.
static constexpr int I2S_SLOTS_PER_FRAME = (AUDIO_SOURCE == AUDIO_SOURCE_PMOD_I2S2) ? 2 : 1;

// ---------------------------------------------------------------------
// Note classes / MIDI range (from class_labels.h)
// ---------------------------------------------------------------------
static const int8_t SILENCE_MIDI_NOTE = -1; // sentinel used in CLASS_TO_MIDI[]

// ---------------------------------------------------------------------
// Inference health target
// ---------------------------------------------------------------------
// Not a gate on anything -- purely the number the boot banner and the periodic health line
// compare the MEASURED Invoke() time against, so a too-slow model announces itself instead of
// quietly degrading note timing.
//
// Why 20 ms: onset timing precision is bounded by the inference period, because the state
// machine cannot know where inside the period the real attack fell. 20 ms of jitter on a note
// start is at the edge of perceptible for percussive material and is comfortably inside what
// a DAW's quantize-on-input will absorb.
static const int TARGET_INFERENCE_PERIOD_MS = 20;

// ---------------------------------------------------------------------
// Note state machine tuning -- ALL IN MILLISECONDS, ALL EVALUATED AGAINST micros().
// See "HOW TIMING WORKS" at the top of this file for why nothing here is converted to a
// frame count at compile time.
// ---------------------------------------------------------------------

// Rolling pitch-vote smoothing window. Votes older than this are dropped from the mode filter.
static const uint32_t MODE_FILTER_WINDOW_MS = 60;

// Minimum time between two Note Ons. LOAD-BEARING, NOT COSMETIC: one pluck produces roughly
// ONSET_BAND_MS worth of consecutive onset-positive windows, so without this, an
// onset-triggered Note On emits one message per positive window -- two or three Note Ons per
// pluck once inference is fast enough to see the whole band.
//
// Sized as the onset band plus one target inference period -- the minimum that fully covers
// the band's last positive window. Do NOT pad this "for safety": every millisecond added is a
// millisecond of genuine fast playing that gets nudged later (held, not dropped -- see the
// lockout branch in runNoteStateMachine).
static const uint32_t RETRIGGER_LOCKOUT_MS =
    (uint32_t)ONSET_BAND_MS + (uint32_t)TARGET_INFERENCE_PERIOD_MS;

// Release safety net #1: if the signal sits below this level continuously for
// RELEASE_HOLD_MS, end the note regardless of what the note head says. Exists because the
// note head can only ever release on a silence CLASSIFICATION -- a decaying string that keeps
// classifying as its own pitch would otherwise leave a Note On hanging with no Note Off, the
// single worst failure mode a MIDI instrument has (audible until a panic message).
//
// -45 dBFS sits just above the -50 dBFS RMS silence gate the training pipeline uses. TUNE
// THIS AGAINST YOUR OWN MIC LEVELS: if notes cut off early during quiet passages, lower it (or
// raise the input gain via MIC_SHIFT_TO_INT16 so the whole signal sits higher); if notes hang
// through silence, raise it.
static const float    RELEASE_RMS_DBFS = -45.0f;
static const uint32_t RELEASE_HOLD_MS  = 120;

// Release safety net #2: a hard ceiling. Nothing rings this long; if a note is still sounding
// after this, something upstream is stuck and a hung note is worse than an early one.
static const uint32_t MAX_NOTE_DURATION_MS = 4000;

// Optional pitch correction while sustaining. 0 = OFF (the default behaviour: Note On
// timing is pinned to the attack and the pitch chosen there is final).
//
// Set to something like 60 to enable: if within this many ms of the Note On the mode filter
// settles on a DIFFERENT pitch, the firmware sends Note Off + a corrected Note On. That
// recovers notes whose first post-attack window was misclassified -- structurally the least
// reliable classification available, since that window is still full of pick transient before
// the harmonic content has developed -- at the cost of emitting an extra note pair into the
// MIDI stream. Try this only if specific notes are landing a semitone or an octave off; it
// does not affect timing either way.
static const uint32_t PITCH_CORRECTION_MS = 0;

static const int MIDI_CHANNEL = 0; // channel 1 in 1-indexed MIDI terms

// ---------------------------------------------------------------------
// DEBUG output level.
//   0 = silent (deployment)
//   1 = events only: every Note On / Note Off / rejected onset, plus a periodic health line.
//       DEFAULT. This is the level that tells you whether timing is right.
//   2 = one line per inference result.
//
// Why 1 and not 2 by default: at the target ~20 ms inference period a level-2 line is ~110
// characters every 20 ms = ~5.5 kB/s = ~55000 baud of the 115200 available, and it is printed
// from audioTask -- the hard real-time task. Once the UART's TX ring buffer fills,
// Serial.printf() BLOCKS the caller, which stalls I2S draining and corrupts exactly the
// timestamps this firmware depends on. If you want level 2, raise SERIAL_BAUD to 921600 below
// and set your serial monitor to match.
// ---------------------------------------------------------------------
#define AUDIO_DEBUG_LEVEL 1
static const unsigned long SERIAL_BAUD = 115200;

// ---------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------
USBMIDI MIDI;
i2s_chan_handle_t rx_chan;

namespace {
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* note_output = nullptr;
TfLiteTensor* onset_output = nullptr;

// Arena size: tune up if AllocateTensors() fails. Left at 160KB even though the current
// (narrow) model needs far less -- it is a small fraction of the S3's 512KB SRAM and
// oversizing costs nothing, whereas undersizing is a boot failure.
constexpr int kTensorArenaSize = 160 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// The op list is read directly off train_guitar_midi.ipynb's build_model(): a Conv2D stem;
// 4 blocks of DepthwiseConv2D + Conv2D(1x1) each followed by MaxPooling2D; then two heads
// that each do a 1x1 Conv2D, a Flatten, and one or two Dense layers (softmax for note,
// sigmoid for onset). BatchNormalization and ReLU are NOT listed separately: the TFLite
// converter folds BatchNorm into the preceding conv's weights and fuses ReLU into that op's
// activation field. AddRelu() is kept as a one-op safety margin in case that fusion doesn't
// apply somewhere. If AllocateTensors() fails with "Didn't find op for builtin opcode 'X'",
// that error names exactly which Add*() call is missing.
tflite::MicroMutableOpResolver<8> resolver;
}  // namespace

// ---------------------------------------------------------------------
// Ping-pong 10ms hop staging buffers ("Buffer A").
// ---------------------------------------------------------------------
int32_t hopBufRaw[2][HOP_SAMPLES * I2S_SLOTS_PER_FRAME]; // raw 32-bit I2S slots
int16_t hopBuf16[2][HOP_SAMPLES];                        // downscaled to mono int16
volatile int activeHopBuf = 0;

// ---------------------------------------------------------------------
// Sliding WINDOW_SAMPLES-sample window ("Buffer B"). Global, not a task-local stack array:
// at 2048 samples this is 4KB.
// ---------------------------------------------------------------------
int16_t slidingWindow[WINDOW_SAMPLES];
int slidingWindowFillCount = 0;

// ---------------------------------------------------------------------
// Inter-core communication.
//
// windowEndMicros is the single most important field in this struct. It is captured the
// instant the hop completing this window comes back from I2S, and is carried untouched
// through inference and into the state machine, so every timing decision is made against when
// the AUDIO happened rather than when the CPU got around to it -- see "HOW TIMING WORKS" above.
// ---------------------------------------------------------------------
struct AudioWindowMsg {
  int16_t  samples[WINDOW_SAMPLES];
  float    hopRmsDbfs;      // RMS (dBFS) of just the newest 10ms hop, for velocity
  uint32_t windowEndMicros; // micros() at the moment this window's last sample arrived
};

struct InferenceResultMsg {
  int      predictedNoteIdx; // argmax of the note softmax head
  float    onsetProb;        // dequantized sigmoid onset probability
  float    hopRmsDbfs;       // passed through from AudioWindowMsg
  uint32_t windowEndMicros;  // passed through from AudioWindowMsg
};

QueueHandle_t audioQueue;   // core0 -> core1, depth 1 (always want the latest window)
QueueHandle_t resultQueue;  // core1 -> core0, depth 1

// ---------------------------------------------------------------------
// Note state machine state (owned by core 0 / audioTask)
//
// There is no separate ATTACK state -- see "HOW TIMING WORKS" at the top of this file for why
// Note On fires directly on the onset instead of behind a stability-count gate.
// ---------------------------------------------------------------------
enum NoteState { STATE_IDLE, STATE_SUSTAIN };
NoteState noteState = STATE_IDLE;

int      sustainingClassIdx  = -1;
int16_t  sustainingMidiNote  = SILENCE_MIDI_NOTE; // int16_t, not uint8_t: SILENCE_MIDI_NOTE is
                                                  // -1 and assigning that into a uint8_t
                                                  // silently wraps to 255.
uint8_t  sustainingVelocity  = 0;
uint32_t noteOnMicros        = 0;  // when the current note's Note On was sent
uint32_t lastNoteOnMicros    = 0;  // for RETRIGGER_LOCKOUT_MS (survives Note Off)
uint32_t quietSinceMicros    = 0;  // start of the current below-RELEASE_RMS_DBFS run, 0 = none
bool     pitchCorrected      = false;

// ---------------------------------------------------------------------
// Time-windowed mode filter.
//
// The ring is generously sized and votes are filtered BY TIMESTAMP at vote time, so the
// smoothing window is MODE_FILTER_WINDOW_MS of real time regardless of inference rate.
// ---------------------------------------------------------------------
static const int MODE_FILTER_CAPACITY = 16;
struct ModeVote { int classIdx; uint32_t atMicros; };
ModeVote modeFilterRing[MODE_FILTER_CAPACITY];
int modeFilterCount = 0;
int modeFilterHead  = 0; // index of the next slot to write

void modeFilterPush(int classIdx, uint32_t atMicros) {
  modeFilterRing[modeFilterHead].classIdx = classIdx;
  modeFilterRing[modeFilterHead].atMicros = atMicros;
  modeFilterHead = (modeFilterHead + 1) % MODE_FILTER_CAPACITY;
  if (modeFilterCount < MODE_FILTER_CAPACITY) modeFilterCount++;
}

// Clears the filter and seeds it with a single class. Called whenever a new onset is accepted,
// so the votes backing the new note reflect the NEW note from its first frame instead of being
// diluted by whatever was sounding (or silent) beforehand.
void modeFilterReset(int seedClassIdx, uint32_t atMicros) {
  modeFilterCount = 0;
  modeFilterHead = 0;
  modeFilterPush(seedClassIdx, atMicros);
}

// Most frequent class among votes newer than MODE_FILTER_WINDOW_MS, ties broken by most recent.
int modeFilterVote(uint32_t nowMicros) {
  static int counts[NUM_NOTE_CLASSES]; // sized exactly, not a fixed 256
  for (int i = 0; i < NUM_NOTE_CLASSES; i++) counts[i] = 0;

  const uint32_t windowUs = MODE_FILTER_WINDOW_MS * 1000UL;

  int best = -1;
  int bestCount = -1;
  // Iterate OLDEST -> NEWEST (not raw buffer-index order, which does not correspond to
  // chronological order once the ring has wrapped), and use `>=` rather than `>` so that among
  // tied counts the class seen LAST wins -- which is what "ties broken by most recent" means.
  for (int n = 0; n < modeFilterCount; n++) {
    int i = (modeFilterHead - modeFilterCount + n + MODE_FILTER_CAPACITY) % MODE_FILTER_CAPACITY;
    // Unsigned subtraction, so this is correct across the ~71 minute micros() wrap.
    if ((uint32_t)(nowMicros - modeFilterRing[i].atMicros) > windowUs) continue;
    int c = modeFilterRing[i].classIdx;
    if (c < 0 || c >= NUM_NOTE_CLASSES) continue;
    counts[c]++;
    if (counts[c] >= bestCount) {
      bestCount = counts[c];
      best = c;
    }
  }
  if (best < 0) best = 0; // class 0 is always "silence" by construction (see class_labels.h)
  return best;
}

// ---------------------------------------------------------------------
// RMS (dBFS) -> MIDI velocity (1-127), linear in dB. Tune MIN/MAX_DBFS to your mic gain.
// ---------------------------------------------------------------------
uint8_t rmsToVelocity(float dbfs) {
  const float MIN_DBFS = -50.0f; // near-silent pluck -> velocity ~1
  const float MAX_DBFS = -6.0f;  // hard pluck -> velocity 127
  float clamped = dbfs;
  if (clamped < MIN_DBFS) clamped = MIN_DBFS;
  if (clamped > MAX_DBFS) clamped = MAX_DBFS;
  float t = (clamped - MIN_DBFS) / (MAX_DBFS - MIN_DBFS);
  int velocity = 1 + (int)roundf(t * 126.0f);
  if (velocity < 1) velocity = 1;
  if (velocity > 127) velocity = 127;
  return (uint8_t)velocity;
}

// ---------------------------------------------------------------------
// I2S setup -- branches at compile time on AUDIO_SOURCE.
// ---------------------------------------------------------------------
void setup_i2s() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);

  // Deeper DMA than the default. This is the buffer that absorbs any moment audioTask is not
  // sitting in i2s_channel_read() -- a Serial.printf(), a state-machine branch, the memmove.
  // If it ever underruns, samples are DROPPED, which does not merely lose audio: it breaks the
  // correspondence between windowEndMicros and the audio it claims to describe, which is the
  // foundation every timing decision in this firmware rests on.
  //
  // dma_frame_num MUST be an exact multiple of HOP_SAMPLES. If it weren't (e.g. a flat 256
  // against a 160-sample hop at 16kHz), the read that drains a freshly-completed descriptor
  // would block on real hardware for most of the hop, but the VERY NEXT read would be
  // satisfied instantly out of the leftover frames still sitting in the driver's ring buffer --
  // no wait on hardware at all. That is indistinguishable, to the backlog detector in
  // audioTask() below, from genuinely falling behind. Sizing dma_frame_num as a multiple of
  // HOP_SAMPLES removes the split entirely: every read either drains a descriptor exactly or
  // blocks for the next one.
  //
  // dma_desc_num x dma_frame_num is sized for >=128ms of slack against the ~10ms service
  // interval.
  chan_cfg.dma_desc_num  = 7;
  chan_cfg.dma_frame_num = HOP_SAMPLES * 2; // exact multiple of the hop size -- see above

  i2s_new_channel(&chan_cfg, NULL, &rx_chan);

#if AUDIO_SOURCE == AUDIO_SOURCE_PMOD_I2S2
  // CS5343: true stereo ADC, needs a real MCLK.
  i2s_std_config_t std_cfg = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = (gpio_num_t)PMOD_MCLK_PIN,
      .bclk = (gpio_num_t)PMOD_BCLK_PIN,
      .ws   = (gpio_num_t)PMOD_WS_PIN,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)PMOD_SD_PIN,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
#else
  // INMP441: mono (L/R tied to GND), derives its own clock from BCLK -- no MCLK needed.
  i2s_std_config_t std_cfg = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)MIC_BCLK_PIN,
      .ws   = (gpio_num_t)MIC_WS_PIN,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)MIC_SD_PIN,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
#endif

  i2s_channel_init_std_mode(rx_chan, &std_cfg);
  i2s_channel_enable(rx_chan);

#if AUDIO_SOURCE == AUDIO_SOURCE_PMOD_I2S2
  Serial.println("Audio source: Pmod I2S2 (3.5mm line-in, CS5343 ADC)");
#else
  Serial.println("Audio source: INMP441 I2S MEMS microphone");
#endif
}

// Blocks until HOP_SAMPLES fresh frames have been read, then downmixes to mono int16 and
// stamps the arrival time.
//
// Returns micros() taken immediately after the read returns. Because i2s_channel_read() blocks
// until the DMA has actually delivered the requested frames, that instant tracks the arrival
// of this hop's LAST sample to within the DMA's fixed granularity -- a constant offset, which
// is all the state machine needs (it reasons about intervals between timestamps, never about
// absolute audio time).
uint32_t read_hop_ping_pong(int bufIdx) {
  size_t bytes_read = 0;
  const size_t want = (size_t)HOP_SAMPLES * I2S_SLOTS_PER_FRAME * sizeof(int32_t);
  i2s_channel_read(rx_chan, hopBufRaw[bufIdx], want, &bytes_read, portMAX_DELAY);
  uint32_t stamp = micros();

  if (bytes_read != want) {
    // Short read. With portMAX_DELAY this should not happen; if it does, the tail of the hop
    // is stale data from the previous pass and the window is no longer contiguous audio.
    // Report rather than silently classify a corrupted window.
    Serial.printf("[I2S] short read: %u of %u bytes\n", (unsigned)bytes_read, (unsigned)want);
  }

#if AUDIO_SOURCE == AUDIO_SOURCE_PMOD_I2S2
  // Stereo capture: average L+R into one mono sample per frame. Correct whether the source is
  // genuinely mono duplicated onto both channels (typical with a mono TS plug into a stereo
  // TRS jack) or a real stereo signal -- either way this matches the mono signal the model was
  // trained on.
  for (int i = 0; i < HOP_SAMPLES; i++) {
    int32_t l = hopBufRaw[bufIdx][2 * i]     >> PMOD_SHIFT_TO_INT16;
    int32_t r = hopBufRaw[bufIdx][2 * i + 1] >> PMOD_SHIFT_TO_INT16;
    int32_t mixed = (l + r) / 2;
    if (mixed > INT16_MAX) mixed = INT16_MAX;
    if (mixed < INT16_MIN) mixed = INT16_MIN;
    hopBuf16[bufIdx][i] = (int16_t)mixed;
  }
#else
  for (int i = 0; i < HOP_SAMPLES; i++) {
    hopBuf16[bufIdx][i] = (int16_t)(hopBufRaw[bufIdx][i] >> MIC_SHIFT_TO_INT16);
  }
#endif
  return stamp;
}

float hop_rms_dbfs(const int16_t* hop, int n) {
  // float, not double: the ESP32-S3 has a single-precision FPU, so double arithmetic here is
  // emulated in software and meaningfully slower for no real benefit.
  float sumSq = 0.0f;
  for (int i = 0; i < n; i++) {
    float s = hop[i] / 32768.0f;
    sumSq += s * s;
  }
  float rms = sqrtf(sumSq / n) + 1e-9f;
  return 20.0f * log10f(rms);
}

// ---------------------------------------------------------------------
// TFLite Micro setup
// ---------------------------------------------------------------------
void setup_model() {
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddLogistic();
  resolver.AddRelu();

  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version mismatch!");
    while (1) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed -- try increasing kTensorArenaSize");
    while (1) delay(1000);
  }

  model_input = interpreter->input(0);

  // Output order matches model.outputs == [note_output, onset_output] in build_model().
  // Sanity-check by SIZE, not by index order: from_keras_model() conversion routinely discards
  // the Keras output names and TFLite's flatbuffer order is not guaranteed to match Keras'.
  // The note head has NUM_NOTE_CLASSES elements and the onset head has exactly 1, which
  // disambiguates them regardless of naming.
  TfLiteTensor* out0 = interpreter->output(0);
  TfLiteTensor* out1 = interpreter->output(1);
  if (out0->dims->data[out0->dims->size - 1] == NUM_NOTE_CLASSES) {
    note_output = out0;
    onset_output = out1;
  } else {
    note_output = out1;
    onset_output = out0;
  }

  Serial.printf("Model loaded. note_output size=%d, onset_output size=%d\n",
                note_output->dims->data[note_output->dims->size - 1],
                onset_output->dims->data[onset_output->dims->size - 1]);
  Serial.printf("Input window: %d samples (%.0fms), hop: %d samples (%.0fms), "
                "norm floor: %.4f, onset threshold: %.2f\n",
                WINDOW_SAMPLES, WINDOW_SAMPLES * 1000.0f / SAMPLE_RATE,
                HOP_SAMPLES, HOP_SAMPLES * 1000.0f / SAMPLE_RATE,
                NORM_PEAK_FLOOR, ONSET_PROB_THRESHOLD);
  Serial.printf("Onset band: %dms (%s) -> retrigger lockout %ums, mode filter %ums\n",
                ONSET_BAND_MS,
                ONSET_BAND_FROM_MODEL ? "from class_labels.h"
                                      : "DEFAULTED -- class_labels.h predates onset-band export",
                (unsigned)RETRIGGER_LOCKOUT_MS, (unsigned)MODE_FILTER_WINDOW_MS);
  Serial.printf("Target inference period: %dms (measured value is printed periodically)\n",
                TARGET_INFERENCE_PERIOD_MS);
}

// Normalizes (zero-mean, peak-scaled against the same floor the training pipeline uses) then
// quantizes a raw int16 window into the model's INT8 input tensor.
//
// THIS MUST MATCH normalize_window_np()/normalize_window_tf() in train_guitar_midi.ipynb
// EXACTLY -- see that notebook's "normalization parity check" cell.
void quantize_and_load_input(const int16_t* window) {
  float sum = 0.0f;
  for (int i = 0; i < WINDOW_SAMPLES; i++) sum += window[i];
  float mean = sum / WINDOW_SAMPLES;

  // static, not a stack-local: at WINDOW_SAMPLES=2048 this is 8KB. One caller (inferenceTask,
  // single-threaded, non-reentrant), so a single reused static instance is safe.
  static float floatWindow[WINDOW_SAMPLES];
  float peak = 0.0f;
  for (int i = 0; i < WINDOW_SAMPLES; i++) {
    float centered = (window[i] - mean) / 32768.0f;
    floatWindow[i] = centered;
    float a = fabsf(centered);
    if (a > peak) peak = a;
  }
  // Divide by max(peak, NORM_PEAK_FLOOR) UNCONDITIONALLY -- see the normalization note near the
  // top of this file.
  float norm = 1.0f / fmaxf(peak, NORM_PEAK_FLOOR);

  float in_scale = model_input->params.scale;
  int in_zero_point = model_input->params.zero_point;

  for (int i = 0; i < WINDOW_SAMPLES; i++) {
    float normalized = floatWindow[i] * norm;
    int32_t q = (int32_t)lroundf(normalized / in_scale) + in_zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    model_input->data.int8[i] = (int8_t)q;
  }
}

bool run_inference(int* outNoteIdx, float* outOnsetProb) {
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    return false;
  }

  int best_idx = 0;
  int8_t best_val = note_output->data.int8[0];
  for (int i = 1; i < NUM_NOTE_CLASSES; i++) {
    if (note_output->data.int8[i] > best_val) {
      best_val = note_output->data.int8[i];
      best_idx = i;
    }
  }
  *outNoteIdx = best_idx;

  float onset_scale = onset_output->params.scale;
  int onset_zero_point = onset_output->params.zero_point;
  *outOnsetProb = (onset_output->data.int8[0] - onset_zero_point) * onset_scale;

  return true;
}

// ---------------------------------------------------------------------
// MIDI helpers -- every Note On/Off in the firmware goes through these two, so there is
// exactly one place that can leave a note hanging, and exactly one place that logs.
// ---------------------------------------------------------------------
void sendNoteOn(int classIdx, uint8_t velocity, uint32_t audioMicros) {
  uint8_t midiNote = (uint8_t)CLASS_TO_MIDI[classIdx];
  MIDI.noteOn(midiNote, velocity, MIDI_CHANNEL);
  sustainingClassIdx = classIdx;
  sustainingMidiNote = (int16_t)midiNote;
  sustainingVelocity = velocity;
  noteOnMicros = micros();
  lastNoteOnMicros = noteOnMicros;
  pitchCorrected = false;
  noteState = STATE_SUSTAIN;
#if AUDIO_DEBUG_LEVEL >= 1
  // "lat" is the end-to-end delay from the audio this decision was made on to the MIDI byte
  // leaving. This is THE number to watch: it should be roughly constant note to note. A stable
  // value (even a biggish one) is fine and a DAW can compensate it; a value that jumps around
  // is what makes playing feel wrong.
  Serial.printf("[NOTE ON ] midi=%3d vel=%3d lat=%lums\n",
                midiNote, velocity, (unsigned long)((micros() - audioMicros) / 1000));
#endif
}

void sendNoteOff(const char* reason) {
  if (noteState != STATE_SUSTAIN) return;
  MIDI.noteOff((uint8_t)sustainingMidiNote, 0, MIDI_CHANNEL);
#if AUDIO_DEBUG_LEVEL >= 1
  Serial.printf("[NOTE OFF] midi=%3d dur=%lums (%s)\n",
                (int)sustainingMidiNote,
                (unsigned long)((micros() - noteOnMicros) / 1000), reason);
#endif
  sustainingMidiNote = SILENCE_MIDI_NOTE;
  sustainingClassIdx = -1;
  noteState = STATE_IDLE;
}

// ---------------------------------------------------------------------
// Note state machine (runs on core 0, once per inference result).
//
//   IDLE    -> SUSTAIN : onset probability crosses AND the voted class isn't silence AND the
//                        retrigger lockout has expired. Note On is sent IMMEDIATELY, so its
//                        timing is pinned to the attack rather than to a separate stability
//                        gate.
//   SUSTAIN -> SUSTAIN : a new onset while already sounding closes the old note and opens a
//                        new one -- INCLUDING at the same pitch, so repeated notes retrigger.
//   SUSTAIN -> IDLE    : voted class is silence, OR the level has been under RELEASE_RMS_DBFS
//                        for RELEASE_HOLD_MS, OR MAX_NOTE_DURATION_MS elapsed. The latter two
//                        are safety nets: a note that never gets a Note Off is worse than one
//                        that ends slightly early.
//
// False-trigger rejection is the mode filter's job (votes over MODE_FILTER_WINDOW_MS of real
// time) and the onset threshold's job, neither of which costs latency. If false triggers
// become a problem, raise MODEL_ONSET_THRESHOLD in the notebook's validation sweep or widen
// MODE_FILTER_WINDOW_MS -- do not add a variable-length delay before Note On; see "HOW TIMING
// WORKS" at the top of this file for why that specifically ruins the feel of playing.
// ---------------------------------------------------------------------
void runNoteStateMachine(const InferenceResultMsg& r) {
  const uint32_t now = micros();

  // RAW vs VOTED -- these are used for DIFFERENT decisions, and conflating them is a bug.
  //
  // At the instant an onset fires, every vote already in the mode filter describes the audio
  // BEFORE the attack: either the previous note, or silence. Judging the new note by that
  // history is not smoothing, it is contamination, and it fails in both directions:
  //   - note after a rest: the filter is full of silence votes, so the voted class IS silence,
  //     so `!isSilence` is false and the onset is REJECTED. The note never sounds at all.
  //   - note after another note: the filter is full of the old pitch, so the new note is
  //     emitted at the OLD pitch.
  //
  // So: the ONSET decision and the Note On pitch use the CURRENT frame's own classification,
  // which is the only evidence that describes the note that just started. The RELEASE decision
  // uses the smoothed vote, where history is genuinely informative because -- thanks to the
  // reset below -- every vote in it belongs to the note currently sounding.
  const int  rawClassIdx  = r.predictedNoteIdx;
  const bool rawIsSilence = (rawClassIdx < 0 || rawClassIdx >= NUM_NOTE_CLASSES) ||
                            (CLASS_TO_MIDI[rawClassIdx] == SILENCE_MIDI_NOTE);

  modeFilterPush(rawClassIdx, r.windowEndMicros);
  const int  votedClassIdx  = modeFilterVote(r.windowEndMicros);
  const bool votedIsSilence = (CLASS_TO_MIDI[votedClassIdx] == SILENCE_MIDI_NOTE);
  const bool onsetFired     = r.onsetProb > ONSET_PROB_THRESHOLD;

  // Track how long the input has been quiet, independent of what the model thinks.
  if (r.hopRmsDbfs < RELEASE_RMS_DBFS) {
    if (quietSinceMicros == 0) quietSinceMicros = r.windowEndMicros;
  } else {
    quietSinceMicros = 0;
  }

  const bool lockoutExpired =
      (lastNoteOnMicros == 0) ||
      ((uint32_t)(now - lastNoteOnMicros) >= RETRIGGER_LOCKOUT_MS * 1000UL);

  // --- New onset: start (or restart) a note. Same path from IDLE and from SUSTAIN. ---
  if (onsetFired && !rawIsSilence) {
    if (!lockoutExpired) {
      // Not dropped, just held: a genuine repeat faster than the lockout will fire on the
      // first frame after the lockout expires, as long as it is still inside its own onset
      // band. So very fast repeats degrade to being nudged later rather than vanishing --
      // bounded by (lockout - repeat interval), and only for repeats above ~12 notes/second.
#if AUDIO_DEBUG_LEVEL >= 2
      Serial.printf("[onset ] held by lockout (%lums since last)\n",
                    (unsigned long)((now - lastNoteOnMicros) / 1000));
#endif
    } else {
      if (noteState == STATE_SUSTAIN) sendNoteOff("retrigger");
      modeFilterReset(rawClassIdx, r.windowEndMicros);
      sendNoteOn(rawClassIdx, rmsToVelocity(r.hopRmsDbfs), r.windowEndMicros);
      return;
    }
  }

  if (noteState != STATE_SUSTAIN) return;

  // --- Optional pitch correction (PITCH_CORRECTION_MS == 0 disables this entirely) ---
  if (PITCH_CORRECTION_MS > 0 && !pitchCorrected && !votedIsSilence &&
      votedClassIdx != sustainingClassIdx &&
      (uint32_t)(now - noteOnMicros) <= PITCH_CORRECTION_MS * 1000UL) {
    uint8_t vel = sustainingVelocity;
    sendNoteOff("pitch correction");
    sendNoteOn(votedClassIdx, vel, r.windowEndMicros);
    pitchCorrected = true;
    return;
  }

  // --- Release paths --- (smoothed vote here: after the reset above, every vote in the filter
  // belongs to the note currently sounding, so history is informative rather than stale.)
  if (votedIsSilence) {
    sendNoteOff("silence");
    return;
  }
  if (quietSinceMicros != 0 &&
      (uint32_t)(r.windowEndMicros - quietSinceMicros) >= RELEASE_HOLD_MS * 1000UL) {
    sendNoteOff("rms release");
    return;
  }
  if ((uint32_t)(now - noteOnMicros) >= MAX_NOTE_DURATION_MS * 1000UL) {
    sendNoteOff("max duration");
    return;
  }
}

// ---------------------------------------------------------------------
// Core 1 task: TFLite Micro inference. Runs flat out, consuming whatever the freshest
// available window is each time it finishes.
// ---------------------------------------------------------------------
volatile uint32_t g_lastInvokeMicros = 0;   // most recent single Invoke() duration
volatile uint32_t g_avgInvokeMicros  = 0;   // rolling average over the last 100 calls

void inferenceTask(void* pvParameters) {
  static AudioWindowMsg windowMsg; // >4KB; static rather than a task-local stack array
  uint32_t inferenceCount = 0;
  uint64_t inferenceMicrosSum = 0;

  for (;;) {
    if (xQueueReceive(audioQueue, &windowMsg, portMAX_DELAY) == pdTRUE) {
      quantize_and_load_input(windowMsg.samples);

      InferenceResultMsg result;
      result.hopRmsDbfs      = windowMsg.hopRmsDbfs;
      result.windowEndMicros = windowMsg.windowEndMicros;

      uint32_t t0 = micros();
      bool ok = run_inference(&result.predictedNoteIdx, &result.onsetProb);
      uint32_t elapsed = micros() - t0;

      g_lastInvokeMicros = elapsed;
      inferenceMicrosSum += elapsed;
      inferenceCount++;
      if (inferenceCount % 100 == 0) {
        g_avgInvokeMicros = (uint32_t)(inferenceMicrosSum / 100);
        inferenceMicrosSum = 0;
      }

      if (ok) {
        // Depth-1 overwrite: only the newest result matters for a real-time stream.
        xQueueOverwrite(resultQueue, &result);
      }
    }
  }
}

// ---------------------------------------------------------------------
// Core 0 task: I2S DMA -> sliding window, RMS, handing windows to core 1, and running the
// note state machine on each new result.
// ---------------------------------------------------------------------
void audioTask(void* pvParameters) {
  // Fill the sliding window completely before starting inference, so the first classification
  // isn't run on a mostly-zero buffer. This also drains whatever the DMA accumulated between
  // i2s_channel_enable() and this task first being scheduled -- important, because reads that
  // return instantly from a backlog would stamp old audio with a current timestamp.
  memset(slidingWindow, 0, sizeof(slidingWindow));
  slidingWindowFillCount = 0;
  while (slidingWindowFillCount < WINDOW_SAMPLES) {
    int buf = activeHopBuf;
    read_hop_ping_pong(buf);
    activeHopBuf ^= 1;

    int n = min(HOP_SAMPLES, WINDOW_SAMPLES - slidingWindowFillCount);
    memcpy(&slidingWindow[slidingWindowFillCount], hopBuf16[buf], n * sizeof(int16_t));
    slidingWindowFillCount += n;
  }

  // static, not a stack-local: at WINDOW_SAMPLES=2048 this struct is >4KB. One producer (this
  // task), copied out by xQueueOverwrite() before the next iteration rewrites it.
  static AudioWindowMsg windowMsg;

  uint32_t lastHopMicros       = micros();
  uint32_t lastHealthPrint     = micros();
  uint32_t backlogHops         = 0;
  uint32_t consecutiveFastHops = 0; // run length of "returned suspiciously fast" reads
  (void)lastHealthPrint; // only read by the health print, which AUDIO_DEBUG_LEVEL 0 compiles out

  for (;;) {
    // 1. Read one 10ms hop. Blocks on real DMA hardware, which is what paces this loop -- no
    //    vTaskDelay needed, and the block is a genuine voluntary yield for anything else
    //    pinned to core 0.
    int buf = activeHopBuf;
    uint32_t hopMicros = read_hop_ping_pong(buf);
    activeHopBuf ^= 1;

    // Backlog detection. If this loop is keeping up, consecutive hops arrive ~HOP_PERIOD_US
    // apart because the read blocks waiting for hardware. If they start arriving much faster,
    // that COULD mean this task fell behind -- draining a backlog -- and every windowEndMicros
    // it produces would be describing audio older than it claims.
    //
    // A SINGLE fast read is not, by itself, evidence of that: with dma_frame_num sized as an
    // exact multiple of HOP_SAMPLES (see setup_i2s()), a stray fast read is ordinary scheduling
    // jitter handing a hop back a little early, not a sign the task is behind. Only TWO OR MORE
    // fast reads IN A ROW is real evidence the loop couldn't keep up: the driver already had a
    // full hop ready when this task came back for the NEXT one too, which only happens if the
    // task spent longer than one hop period doing something else (a printf, a MIDI send, the
    // memmove). Silent when healthy.
    if ((uint32_t)(hopMicros - lastHopMicros) < (uint32_t)(HOP_PERIOD_US / 2)) {
      consecutiveFastHops++;
      if (consecutiveFastHops >= 2) backlogHops++;
    } else {
      consecutiveFastHops = 0;
    }
    lastHopMicros = hopMicros;

    // 2. Slide the window: drop the oldest HOP_SAMPLES, append the newest.
    memmove(slidingWindow, slidingWindow + HOP_SAMPLES,
            (WINDOW_SAMPLES - HOP_SAMPLES) * sizeof(int16_t));
    memcpy(slidingWindow + (WINDOW_SAMPLES - HOP_SAMPLES), hopBuf16[buf],
           HOP_SAMPLES * sizeof(int16_t));

    float hopRms = hop_rms_dbfs(hopBuf16[buf], HOP_SAMPLES);

    // 3. Offer the fresh window to core 1 EVERY hop. There is no decimation: the depth-1
    //    overwrite queue already self-throttles to whatever rate inference can sustain, and
    //    offering every hop means the window inference picks up is never more than one hop
    //    stale.
    memcpy(windowMsg.samples, slidingWindow, sizeof(slidingWindow));
    windowMsg.hopRmsDbfs      = hopRms;
    windowMsg.windowEndMicros = hopMicros;
    xQueueOverwrite(audioQueue, &windowMsg);

    // 4. Consume the latest inference result, if one has appeared.
    InferenceResultMsg result;
    if (xQueueReceive(resultQueue, &result, 0) == pdTRUE) {
#if AUDIO_DEBUG_LEVEL >= 2
      int predMidi = (result.predictedNoteIdx >= 0 && result.predictedNoteIdx < NUM_NOTE_CLASSES)
                         ? CLASS_TO_MIDI[result.predictedNoteIdx] : -2;
      const char* stateName = (noteState == STATE_IDLE) ? "IDLE" : "SUSTAIN";
      Serial.printf("[AUDIO] rms=%.1f cls=%d midi=%d onset=%.2f/%.2f age=%lums %s\n",
                    result.hopRmsDbfs, result.predictedNoteIdx, predMidi, result.onsetProb,
                    ONSET_PROB_THRESHOLD,
                    (unsigned long)((micros() - result.windowEndMicros) / 1000), stateName);
#endif
      // Advance the state machine ONLY on a genuinely new classification. Calling it every
      // loop iteration (~10ms) would feed modeFilterPush() the same stale result repeatedly,
      // starving the vote of genuinely different opinions.
      runNoteStateMachine(result);
    }

    // 5. Periodic health line. This is the one number that tells you whether the model fits
    //    the timing budget -- printed even at AUDIO_DEBUG_LEVEL 1 because of how directly
    //    every timing symptom in this firmware traces back to it.
#if AUDIO_DEBUG_LEVEL >= 1
    if ((uint32_t)(micros() - lastHealthPrint) >= 5000000UL) {
      lastHealthPrint = micros();
      uint32_t avgUs = g_avgInvokeMicros;
      if (avgUs > 0) {
        Serial.printf("[HEALTH] Invoke avg %luus (%.1fx the %dms target)%s\n",
                      (unsigned long)avgUs,
                      avgUs / (TARGET_INFERENCE_PERIOD_MS * 1000.0f),
                      TARGET_INFERENCE_PERIOD_MS,
                      (avgUs > (uint32_t)TARGET_INFERENCE_PERIOD_MS * 1000UL)
                          ? "  <-- TOO SLOW: onsets are being sampled past; shrink the model"
                          : "  OK");
      }
      if (backlogHops > 0) {
        Serial.printf("[HEALTH] audioTask fell behind on %lu hop(s) -- timestamps unreliable\n",
                      (unsigned long)backlogHops);
        backlogHops = 0;
      }
    }
#endif
  }
}

// ---------------------------------------------------------------------
// DEBUG: MIDI heartbeat. Sends a test Note On, then a Note Off one second later, forever,
// completely independent of the audio/inference pipeline's DATA. Answers exactly one question
// in isolation -- "does a MIDI message sent by this firmware actually reach the DAW?" -- before
// spending time debugging pitch detection, onset thresholds, or wiring. Off by default; flip
// MIDI_DEBUG_HEARTBEAT to 1 only when MIDI transport itself needs testing in isolation.
//
// MUST RUN AS ITS OWN PINNED TASK ON CORE 0, NOT FROM ARDUINO'S loop(). loop() runs as a
// priority-1 task pinned to core 1 alongside inferenceTask (priority 2). inferenceTask has
// fresh audio waiting the instant Invoke() returns, so it never voluntarily blocks for long --
// and a task that never yields starves any lower-priority task sharing its core. Pinning this
// heartbeat to core 0 instead, where audioTask spends most of its time genuinely blocked inside
// i2s_channel_read() on real DMA hardware, gives it real scheduling opportunities.
// ---------------------------------------------------------------------
#define MIDI_DEBUG_HEARTBEAT 0

#if MIDI_DEBUG_HEARTBEAT
void midiDebugHeartbeatTask(void* pvParameters) {
  bool noteIsOn = false;
  const uint8_t DEBUG_NOTE = 60;      // middle C
  const uint8_t DEBUG_VELOCITY = 100;
  for (;;) {
    if (!noteIsOn) {
      MIDI.noteOn(DEBUG_NOTE, DEBUG_VELOCITY, MIDI_CHANNEL);
      Serial.printf("[MIDI DEBUG] Note ON  note=%d vel=%d ch=%d\n",
                    DEBUG_NOTE, DEBUG_VELOCITY, MIDI_CHANNEL + 1);
    } else {
      MIDI.noteOff(DEBUG_NOTE, 0, MIDI_CHANNEL);
      Serial.printf("[MIDI DEBUG] Note OFF note=%d\n", DEBUG_NOTE);
    }
    noteIsOn = !noteIsOn;
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
#endif

void setup() {
  Serial.begin(SERIAL_BAUD);

  USB.begin();
  MIDI.begin();

  setup_i2s();
  setup_model();

  audioQueue = xQueueCreate(1, sizeof(AudioWindowMsg));
  resultQueue = xQueueCreate(1, sizeof(InferenceResultMsg));

  // Task stacks: 16KB each. The large per-call buffers (quantize_and_load_input()'s float
  // scratch window, both AudioWindowMsg instances) are all `static`, so this is headroom for
  // TFLite Micro's own per-op dispatch rather than a number chosen to just barely fit.
  static const uint32_t TASK_STACK_BYTES = 16384;

  xTaskCreatePinnedToCore(inferenceTask, "inferenceTask", TASK_STACK_BYTES, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(audioTask,     "audioTask",     TASK_STACK_BYTES, NULL, 2, NULL, 0);

#if MIDI_DEBUG_HEARTBEAT
  xTaskCreatePinnedToCore(midiDebugHeartbeatTask, "midiHeartbeat", 4096, NULL, 1, NULL, 0);
#endif

  Serial.println("Guitar-to-MIDI ready.");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}

/*
 * ---------------------------------------------------------------------
 * OPERATING NOTES / KNOWN LIMITATIONS
 * ---------------------------------------------------------------------
 * 1. WATCH THE [HEALTH] LINE FIRST, ALWAYS. If it says TOO SLOW, no amount of threshold or
 *    gate tuning will fix note dropouts -- the onset peak is not being sampled at all. Shrink
 *    the model (MODEL_WIDTH / STEM_STRIDE in train_guitar_midi.ipynb) until that line says OK,
 *    then tune.
 * 2. Note On latency is fixed rather than variable, but it is not zero: it is roughly one
 *    inference period plus the USB MIDI hop. Watch the "lat=" figure in the [NOTE ON] lines --
 *    what matters is that it is CONSISTENT note to note, not that it is small. A constant
 *    offset is trivially compensated by any DAW's track delay; a varying one is not
 *    compensable by anything.
 * 3. MIC_SHIFT_TO_INT16 / PMOD_SHIFT_TO_INT16 set each input path's effective gain. If
 *    rmsToVelocity() pins near the bottom of its range during normal playing at a normal
 *    distance, lower the relevant shift rather than compressing rmsToVelocity()'s range.
 * 4. RETRIGGER_LOCKOUT_MS is derived from the onset band width, which arrives via
 *    class_labels.h's MODEL_ONSET_WINDOW_MS. Older class_labels.h files predating that export
 *    fall back to 40ms and say "DEFAULTED" at boot -- regenerate model.h/class_labels.h from
 *    the current notebook to pick up the real value.
 * 5. `resolver`'s op list was derived by reading build_model() in train_guitar_midi.ipynb, not
 *    by inspecting the converted .tflite. If AllocateTensors() ever fails with "Didn't find op
 *    for builtin opcode 'X'", add the corresponding resolver.AddX() call and bump
 *    MicroMutableOpResolver<N>'s N.
 * 6. There is no FFT/mel code to keep in sync -- the only shared state between this file and
 *    the training pipeline is the constants pulled from class_labels.h at the top.
 * 7. The dma_frame_num-is-a-multiple-of-HOP_SAMPLES invariant in setup_i2s() assumes a 16 kHz
 *    / 160-sample hop. If HOP_SAMPLES ever changes (a different notebook config),
 *    dma_frame_num = HOP_SAMPLES * 2 stays an exact multiple automatically since it's computed
 *    from the same constant -- nothing to update by hand. A nonzero but FLAT "fell behind"
 *    count in [HEALTH] is expected jitter; a GROWING one is a real backlog -- look for
 *    something on core 0 blocking for longer than a hop period (a printf, a slow MIDI call,
 *    USB stalls).
 */
