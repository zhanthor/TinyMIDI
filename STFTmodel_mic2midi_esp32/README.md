# MIDI note classifier — ESP32 deployment

Port of the EE 446 final-project model to an ESP32 with an I2S MEMS
microphone, following the Lab 5 deployment workflow.

## What carried over from Lab 5, and what didn't

| Lab 5 file | Status here |
|---|---|
| `recognize_commands.{h,cpp}` | Reimplemented as `note_recognizer.{h,cpp}` — same averaging + suppression logic, 33 classes, float scores |
| `micro_speech.ino`, `arduino_main.cpp`, `main_functions.h` | Collapsed into `midi_note_esp32.ino` |
| `micro_features_micro_model_settings.{h,cpp}` | Replaced by generated `model_params.h` |
| `micro_features_model.{h,cpp}` | Replaced by generated `model.h` |
| `arduino_audio_provider.cpp` | **Discarded.** Nano 33 BLE PDM library; no Xtensa equivalent. Rewritten as `audio_provider.cpp` on I2S |
| `micro_features_micro_features_generator.{h,cpp}` | **Discarded.** Produces 40-bin log-Mel microfrontend features; this model wants a 217-bin log-*frequency* (musical) representation. Rewritten as `feature_provider.cpp` |
| `sparkfun_edge_*.cpp` | Not applicable |
| `micro_features_{yes,no}_micro_features_data.*` | Test fixtures for the speech model; replaced by `golden_test.h` |

## Build

1. **Board support** — Arduino IDE → Preferences → Additional Board Manager
   URLs → `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`,
   then Boards Manager → install **esp32**.
2. **TFLM library** — Library Manager → install **`tflm_esp32`**.
   (Espressif's `esp-tflite-micro` + `esp-nn`, packaged for Arduino. Lab 5's
   `Arduino_TensorFlowLite` is Cortex-M only and will not compile here.)
3. **Generate headers** — run notebook section 9, then copy `model.h`,
   `model_params.h`, `filterbank.h` and `golden_test.h` into this folder.
   All four are regenerated together; mixing versions will fail the golden test.
4. **Board settings**
   - Board: **ESP32 Dev Module**
   - Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)** — the model is now
     only ~24 KB, but the TFLM runtime itself is still large; keep this setting
     unless you have measured otherwise
   - CPU Frequency: **240 MHz**
5. Compile and upload. Serial Monitor at **115200**.

## Wiring

```
INMP441      ESP32
-------      -----
VDD    ->    3V3        (not 5V — 5V will damage the module)
GND    ->    GND
SCK    ->    GPIO32     (BCLK)
WS     ->    GPIO25     (LRCL)
SD     ->    GPIO33     (DOUT)
L/R    ->    GND        (selects the left slot)

MIDI DIN OUT               ESP32
-------------              -----
Pin 1    ->            Not Connected    
Pin 2    ->                 GND    
Pin 3    ->            Not Connected
Pin 4    ->  222 ohm ->     3V3   
Pin 5    ->  222 ohm ->   GPIO17/TX2 
```

Pins are `#define`d at the top of `audio_provider.cpp`.

**SPH0645 users:** the ESP32's I2S peripheral latches this part on the wrong
clock edge in some core versions. If you get plausible-looking but consistently
wrong output, either switch to an INMP441 or apply the known `I2S_TIMING_REG` /
`bit_shift` workaround for your core version.

## Bring-up order

Do these in sequence. Each step isolates one failure mode.

**1.(Disabled) Golden test.** `ENABLE_GOLDEN_TEST` is on by default. On boot you should
see both `STFT ... PASS` and `Inference ... PASS`. This runs entirely on
canned data with no microphone involved, so it separates "the port is wrong"
from "the audio is wrong" — which is the distinction that costs the most time
if you skip it.

- *STFT fails* → the Hann window or frame geometry disagrees with the
  notebook. The usual culprit is a symmetric Hann (`cos(2πn/(N-1))`) instead
  of the periodic one TensorFlow uses (`cos(2πn/N)`).
- *STFT passes, inference fails* → schema or op mismatch. Check you exported
  without `EXPERIMENTAL_SPARSITY`.

**2. Microphone gain — now just a sanity check.** The log-frequency features
are gain-invariant by construction (measured drift under 0.0004% across a 400×
level range), so `kMicGain` no longer needs calibrating. Uncomment the
diagnostic `Serial.printf` and confirm only that `rms` clears `kRmsGate` when
a note is playing. If it doesn't, raise `kMicGain` until it does.

**3. Timing.** In the same diagnostic line, confirm `feat` + `infer` stay well
under 32 ms. There is now one 2048-point FFT instead of 29 128-point ones, and
the model is ~10× smaller, so there should be ample headroom — `kHopSamples`
can drop to 256 (16 ms) for a faster update rate.

Be clear about what that does and does not fix: the hop controls how often you
get a new estimate, but a freshly struck note is not fully inside the 128 ms
window until 128 ms have elapsed. Shrinking the hop will not beat that floor.

**4. Live notes.** Re-comment the diagnostic and play notes. Output format
matches Lab 5: `Heard 60 (198) @14320ms`.

## Tuning

| Constant | File | Effect |
|---|---|---|
| `kMicGain` | `audio_provider.cpp` | Input level. Only needs to clear the RMS gate. |
| `kHopSamples` | `.ino` | Inference rate. 512 = 32 ms = ~31/sec |
| `kRmsGate` | `.ino` | Silence cutoff |
| `kTensorArenaSize` | `.ino` | Trim to the reported `arena_used_bytes` + ~10% |
| detection threshold | `note_recognizer.h` ctor default | Raise for fewer false notes, lower for faster response |
| suppression window | `note_recognizer.h` ctor default | Minimum gap before the same note repeats |

## Window length and latency

The clip is 128 ms (2048 samples). That sets a frequency resolution limit of
7.8 Hz, against a 4.9 Hz gap between adjacent semitones at the bottom of the
range — so the fundamental is still not directly resolvable at MIDI 40, and the
model reads the harmonic series to disambiguate. The log-frequency axis is what
makes that tractable.

Measured accuracy of a linear probe on the features:

| window | all 32 notes | low register | low + dull source |
|---|---|---|---|
| 64 ms | 96.2% | 92.6% | 82.8% |
| **128 ms (current)** | **98.9%** | **98.0%** | **95.8%** |
| 256 ms | 97.6% | 99.6% | 98.8% |

128 ms is the knee. Going to 256 ms adds a few points on dull low notes but
doubles detection latency, doubles FFT RAM to 56 KB, and starts straddling
sixteenth notes at 120 BPM. It also *loses* accuracy on the all-notes column,
because a long window on a decaying note spends its tail in low-SNR silence.

**Detection latency equals the window length**, not the hop. Shrinking
`kHopSamples` raises the update rate but cannot beat the 128 ms floor.

## If you need more accuracy

The remaining levers are in the notebook, not here:

- `AUDIO_LENGTH` → 4096 if your notes are sustained and 256 ms latency is
  acceptable. Set `FFT_SIZE` to 4096 as well, and raise `kFFTMaxSize` in
  `fft.h` to 4096 (FFT RAM goes to 56 KB).
- `BINS_PER_SEMI` → 4 for finer pitch quantization (measured as roughly neutral
  on synthetic data; worth trying on yours).
- `MIDI_HIGH` upward if your source has useful energy above 4186 Hz.

The firmware picks up all of these automatically from the regenerated headers —
the only manual edit is `kFFTMaxSize` when `FFT_SIZE` exceeds 2048.
