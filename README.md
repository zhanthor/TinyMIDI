# MIDI note classifier — Arduino Nano 33 BLE Sense

Second deployment target for the EE 446 final project, alongside the ESP32
build. Same model, same feature front end, different capture and output.

## Board requirement

Must be the **Nano 33 BLE Sense** (rev 1 or rev 2). The plain **Nano 33 BLE has
no microphone** and will compile but capture nothing.

## What is shared with the ESP32 build, and what is not

| File | Status |
|---|---|
| `model.h`, `model_params.h`, `filterbank.h`, `golden_test.h` | **Identical.** Generated once by the notebook, used by both boards unchanged |
| `feature_provider.{h,cpp}` | Same maths; adds an optional CMSIS-DSP path |
| `fft.{h,cpp}`, `note_recognizer.{h,cpp}` | Byte-identical to the ESP32 build |
| `audio_provider.{h,cpp}` | **Rewritten** — PDM instead of I2S |
| `midi_note_nano33.ino` | **Rewritten** — benchmark mode, BLE MIDI, note names |
| `ble_midi.{h,cpp}` | **New** — MIDI-over-BLE output |

That the model headers transfer with no changes is worth a sentence in the
report: the front end is defined entirely by the generated constants, so
retargeting is a capture-and-output problem, not a model problem.

## Setup

1. **Boards Manager** → *Arduino Mbed OS Nano Boards*
2. **Library Manager** → **`Chirale_TensorFlowLite`**, and **delete
   `Arduino_TensorFlowLite` if you have it.** Do not keep both — see below.
3. **Library Manager** → `ArduinoBLE`
4. *(optional)* **Library Manager** → `Arduino_CMSIS-DSP`, then set
   `USE_CMSIS_DSP 1` in `feature_provider.cpp`
5. Copy the four generated headers from the notebook into this folder.
6. Board: **Arduino Nano 33 BLE**. Serial Monitor at **115200**.

## If the Serial Monitor shows nothing

This is the most common first symptom on this board, and it is usually **not** a
problem with the model.

**The Nano 33 BLE does not reset when you open the Serial Monitor.** It starts
running the instant it is powered, so boot messages are printed before the
monitor attaches and you see an empty window — which looks exactly like a
crashed board. The sketch now waits up to 15 seconds for the monitor
(`kSerialWaitMs`), then continues anyway so the device still works standalone.

**Read the RGB LED first.** It tells you what the board is doing with no Serial
Monitor attached:

| LED | Meaning |
|---|---|
| **Blue blinking** | Waiting for the Serial Monitor — open it now |
| **Green solid** | Running normally |
| **Red, 2 blinks** | Feature geometry mismatch (`model_params.h` / `filterbank.h` disagree) |
| **Red, 3 blinks** | Model schema mismatch — regenerate `model.h` |
| **Red, 4 blinks** | `AllocateTensors` failed — arena too small, or `EXPERIMENTAL_SPARSITY` |
| **Red, 5 blinks** | Model input shape disagrees with `model_params.h` |
| **Red, 6 blinks** | PDM init failed — wrong board variant? |
| **Nothing at all** | The sketch is not running. See the checklist below |

Once running, a `[alive]` heartbeat prints every 5 seconds with the current RMS,
so "a quiet room" and "a hung loop" are no longer indistinguishable.

### Checklist when the LED is completely dark

1. **Run `nano33_selftest` first.** It has no TFLM, no model, and no generated
   headers. If it works, the board, USB, and microphone are all fine and the
   problem is in the main sketch. If it does not, the problem is upstream of
   anything model-related. This single step saves the most time.
2. **Re-select the port.** The board re-enumerates after upload, so
   *Tools → Port* may now point at nothing. Pick it again.
3. **Baud must be 115200** in the Serial Monitor dropdown.
4. **Board must be *Arduino Nano 33 BLE*** under *Tools → Board → Arduino Mbed
   OS Nano Boards*. The self-test prints a warning if it was compiled for
   anything else.
5. **Double-tap the reset button.** The orange LED should pulse slowly, meaning
   the bootloader is running. If it does, the board is alive and the sketch is
   crashing — re-upload the self-test.
6. **Try a different USB cable.** Charge-only cables enumerate nothing. A
   surprising number of "dead board" reports are this.

### If the self-test works but the main sketch does not

Then it is the sketch, and the LED code identifies the stage. The most likely
causes, in order:

- **Missing or mismatched headers.** All four of `model.h`, `model_params.h`,
  `filterbank.h`, `golden_test.h` must come from the *same* notebook run.
  Mixing versions gives 2 or 5 red blinks.
- **Arena too small** (4 blinks) — raise `kTensorArenaSize` to 48 KB and retry.
- **RAM exhaustion.** Unlike the arena, this shows up as a dark board rather than
  a clean error, because the crash happens during startup. If the compile output
  reports high global usage, set `ENABLE_BLE_MIDI 0` to free ~15 KB and see
  whether it boots.


## Measured on hardware (Nano 33 BLE Sense, 64 MHz)

```
Arena used         : 6,724 / 32,768 bytes
Golden test        : PASS  (feature error 0.00096% of range, prediction correct)
Feature extraction : 17.53 ms
Inference          : 116.05 ms
TOTAL per frame    : 133.57 ms   -> ~7.5 inferences/s
```

The arena figure is the one to report as RAM. Note it came in at 6.7 KB, far
under the 32 KB allocated — you can trim `kTensorArenaSize` to 12 KB.

**Inference at 116 ms is 6.8 cycles per MAC.** I originally read that as TFLM's
reference kernels, but that was wrong: inspecting both library source trees shows
**both ship the CMSIS-NN accelerated kernels** (`kernels/cmsis_nn/conv.cpp`
calling `arm_convolve_wrapper_s8`, plus `fully_connected`, `pooling`, `softmax`)
and **neither ships a reference `conv.cpp` at all**. So slow kernels cannot be
the explanation. See "Which TFLM library" below for what to check instead.

### Does it still classify guitar into the Serial Monitor?

Yes. `kHopSamples` is 2048 (128 ms, back-to-back windows), the recognizer counts
frames rather than milliseconds, and notes print as `Heard A2 (45) conf 0.87`.
End-to-end you should see a note about **400 ms** after plucking it: 128 ms to
fill the window, plus two frames to confirm.

That is fine for demonstrating individual notes. It is too slow to track a fast
passage, and too slow for BLE MIDI to feel musical.

### Getting it faster, in order of effort

| Change | Inference | Notes |
|---|---|---|
| **`USE_CMSIS_DSP 1`** | unchanged | Cuts *feature* time 17.5 → ~5 ms. Two-minute change, golden test verifies it |
| Trim `kTensorArenaSize` to 12 KB | unchanged | Frees ~20 KB RAM |
| Remove the duplicate TFLM library | possibly 116 → ~30 ms | Free; check for "Multiple libraries were found" first |
| Retrain with 8 filters in conv2/conv3 | 116 → ~46 ms | 435 k MACs. **Keeps the full 2.8-octave receptive field** |
| Retrain with 8 filters *and* 13 taps | 116 → ~29 ms | 276 k MACs, but receptive field drops to 1.7 octaves |

If you retrain, cut **filters** before **kernel width**. At 128 ms the pitch
information lives in harmonics 2–3 octaves above the fundamental, so a receptive
field under ~2 octaves removes what the model actually relies on. Narrowing the
kernel is the change most likely to cost accuracy.

### Recommendation for the demo

**Use the ESP32 as the primary demo target** (≈11 ms/frame, 31 inferences/s) and
**move BLE MIDI there** — the ESP32 has BLE too, and at 11 ms per frame it feels
like an instrument rather than a delayed transcription. Keep the Nano as the
hardware-comparison row in your report: same model, same features, same headers,
3.75× slower core, and a measured latency difference to explain. That contrast
is a genuine result, and it is more interesting than either board alone.


## Which TFLM library — and why not both

**Install `Chirale_TensorFlowLite` and delete `Arduino_TensorFlowLite`.**

### Having both installed is the actual hazard

Both libraries provide the same `tensorflow/lite/...` header paths that this
sketch includes. With both present, the Arduino builder must guess which library
satisfies those includes — and it can resolve **headers from one library while
linking objects from the other**. That is an ODR violation: it may compile
cleanly and then behave, or perform, unpredictably.

**Check your compile output for `Multiple libraries were found for ...`.** If it
is there, that is exactly what happened, and it is a plausible contributor to the
116 ms. Removing the duplicate costs nothing and is the first thing to try.

Library Manager cannot uninstall `Arduino_TensorFlowLite` because Arduino
delisted it, so delete the folder manually:

```
Windows   %USERPROFILE%\Documents\Arduino\libraries\Arduino_TensorFlowLite
macOS     ~/Documents/Arduino/libraries/Arduino_TensorFlowLite
Linux     ~/Arduino/libraries/Arduino_TensorFlowLite
```

Then restart the IDE and re-run the benchmark.

### Why Chirale is the one to keep

| | Chirale_TensorFlowLite | Arduino_TensorFlowLite |
|---|---|---|
| Version | 2.0.0, maintained | 2.4.0-**ALPHA** (2021), delisted |
| In Library Manager | yes | no |
| `architectures=` | `mbed_nano, esp32, mbed_nicla, mbed_portenta, mbed_giga` | *(unrestricted)* |
| Build model | **compiles from source** | `precompiled=full` — links a prebuilt `.a` |
| CMSIS-NN kernels | yes | yes |
| Reference conv kernel | not shipped | not shipped |

The build model is the deciding factor. Chirale compiles in your toolchain, so
the flags are visible and changeable — including whether CMSIS-NN gets its
Cortex-M4 DSP fast path (`ARM_MATH_DSP`), which is worth roughly 4× on the inner
loop. `Arduino_TensorFlowLite` links a binary built in 2021 with flags you cannot
inspect.

### If 116 ms persists after removing the duplicate

Then CMSIS-NN is being used but without its DSP path, or the `(21,1)` kernel
shape is hitting a slow dispatch inside `arm_convolve_wrapper_s8`. At that point
the reliable fix is on the model side, not the library side — retrain with 8
filters in conv2/conv3 (~46 ms, keeps the full receptive field), per the table
above. That change is entirely in your control and does not depend on library
internals.

## Bring-up

The sketch runs three checks at boot, in order. Read them before anything else.

**1. Golden test.** Replays a canned clip through the whole pipeline and
compares against the notebook. Expect `Features ... PASS` and
`Inference ... PASS`. The portable FFT path has been verified against the Python
reference to 0.0002% of range; if you enable `USE_CMSIS_DSP` and this starts
failing, the real-FFT output packing is the first thing to check.

**2. Arena report.** `Arena used: N / 32768 bytes` — this is the RAM figure for
your slide.

**3. Latency benchmark.** 20 iterations of feature extraction and inference,
reporting mean and max. **These are the on-device latency numbers to report.**
It also warns if processing exceeds 80% of the hop budget.

Estimated for reference — measure, don't quote these:

| configuration | FFT | inference | total |
|---|---|---|---|
| ESP32 @ 240 MHz | ~1.5 ms | ~9 ms | ~11 ms |
| Nano 33 @ 64 MHz, portable | ~5.5 ms | ~100 ms | ~106 ms |
| Nano 33 @ 64 MHz, CMSIS | ~1.8 ms | ~21 ms | ~23 ms |

The spread on inference is enormous because it depends on whether the TFLM build
uses CMSIS-NN optimised int8 kernels or the portable reference ones. If the
benchmark reports ~100 ms, that is almost certainly why.

## Why the hop is longer here

`kHopSamples = 1024` (64 ms), against 512 (32 ms) on the ESP32. At 64 MHz one
inference can outlast a 32 ms hop, and a pipeline that cannot keep up
accumulates lag until it is reporting notes from seconds ago.

The audio provider handles this explicitly: on overrun it **discards the backlog
and keeps the newest audio**, so latency stays bounded and frames are skipped
instead. `AudioProviderOverruns()` counts what was dropped — if it climbs
steadily, raise `kHopSamples`.

The hop sets the **update rate only**. Detection latency is set by the 128 ms
window and is identical on both boards.

## BLE MIDI

`ENABLE_BLE_MIDI 1` advertises the standard MIDI-over-BLE service, so the board
becomes a wireless MIDI controller: play the guitar, and a soft-synth on a
laptop or phone plays the detected notes.

Pairing:
- **macOS** — Audio MIDI Setup → Window → Show MIDI Studio → Bluetooth → Connect
- **iOS** — any BLE-MIDI host app
- **Windows** — a BLE MIDI bridge plus loopMIDI
- **Linux** — bluez with the BLE-MIDI ALSA bridge

Note-off is driven by the *smoothed* label rather than by new-detection events,
so a held note is released when the pitch changes or confidence drops. Without
that the synth would drone indefinitely.

**Rename the device before your demo.** `kDeviceName` in `ble_midi.h` is
`"TinyMIDI-Guitar"`; the presentation guidelines warn that many BLE devices will
be active in the room. Make it unique and verify pairing beforehand.

## Suggested comparison table for the report

Running both boards gives a hardware-tradeoff row that most projects will not
have. Fill in from each board's boot output:

| | ESP32 | Nano 33 BLE Sense |
|---|---|---|
| Core | Xtensa LX6 @ 240 MHz | Cortex-M4F @ 64 MHz |
| RAM total | 520 KB | 256 KB |
| Flash total | 4 MB | 1 MB |
| Microphone | I2S INMP441 (external) | PDM MP34DT05 (onboard) |
| Arena used | *(boot print)* | *(boot print)* |
| Flash used | *(compile output)* | *(compile output)* |
| Feature latency | *(benchmark)* | *(benchmark)* |
| Inference latency | *(benchmark)* | *(benchmark)* |
| Update rate | 31 Hz (32 ms hop) | 15.6 Hz (64 ms hop) |
| Detection latency | 128 ms | 128 ms |
| Output | Serial | Serial + BLE MIDI |

The 27.8 KB model matters most on this board: the original 230 KB version would
have consumed a quarter of the Nano's flash before the TFLM runtime and mbed
core were linked in.

## Troubleshooting

| Symptom | Cause |
|---|---|
| **No serial output at all** | Monitor opened after boot — see the section above; the board does not reset on connect |
| **Board dark, no LED** | Sketch not running — run `nano33_selftest`, re-select the port, try another cable |
| `PDM init failed` | Non-Sense board, or PDM already in use |
| Golden test FAILs on features | `USE_CMSIS_DSP` mismatch, or headers regenerated inconsistently — always copy all four together |
| `AllocateTensors failed` | Raise `kTensorArenaSize`; or the model was exported with `EXPERIMENTAL_SPARSITY` |
| Inference ~100 ms | TFLM built without CMSIS-NN kernels |
| Overruns climbing | Hop shorter than processing time — raise `kHopSamples` |
| Notes drone on the synth | BLE note-off not delivered; check `kMidiThreshold` isn't so high that the label never updates |
| Predictions flicker | Raise the detection threshold or history in `note_recognizer.h` |
