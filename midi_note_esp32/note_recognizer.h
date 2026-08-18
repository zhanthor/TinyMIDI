// note_recognizer.h -- posterior smoothing over a sliding window of results.
//
// Functionally this is Lab 5's recognize_commands.{h,cpp}, reimplemented for a
// many-class pitch model with float scores. A single inference is noisy, so we
// average class scores over a short history, require the winner to clear a
// confidence threshold, and suppress repeats until the note changes or a
// refractory period expires.
//
// IMPORTANT DESIGN NOTE: the history window is counted in FRAMES, not
// milliseconds. An earlier version used wall-clock ms, which breaks silently on
// slow hardware: if one inference takes longer than the history window, fewer
// frames land inside it than min_count requires, Update() bails out every time,
// and the device reports nothing at all while looking perfectly healthy. Frame
// counting is invariant to how fast the board happens to be.
//
// max_age_ms still discards genuinely stale frames, so a long stall does not
// average across a gap in time.

#pragma once

#include <stdint.h>

#include "model_params.h"

class NoteRecognizer {
 public:
  // history_frames   : how many recent inferences to average (not milliseconds)
  // detection_thresh : minimum averaged score (0-1) for the winner
  // suppression_ms   : minimum gap before the SAME note reports again
  // min_count        : minimum frames present before reporting; must be <=
  //                    history_frames or nothing is ever reported
  // max_age_ms       : frames older than this are ignored
  NoteRecognizer(int history_frames = 4,
                 float detection_thresh = 0.55f,
                 int32_t suppression_ms = 250,
                 int min_count = 2,
                 int32_t max_age_ms = 2000);

  // Re-tunes the recognizer after construction and clears the history.
  //
  // This exists because the useful values are not all knowable at compile time.
  // history_frames is: counting frames is deliberately independent of how fast
  // the board runs. But suppression_ms and max_age_ms are wall-clock, so they
  // have to be sized against the MEASURED frame time -- a suppression window
  // shorter than one frame would let a held note retrigger on every frame.
  void Configure(int history_frames,
                 float detection_thresh,
                 int32_t suppression_ms,
                 int min_count,
                 int32_t max_age_ms = 2000);

  // Feeds one dequantized score vector (length kCategoryCount, summing to ~1).
  // Sets *label / *score to the smoothed winner and *is_new to true only on a
  // fresh detection worth printing.
  void Update(const float* scores,
              int32_t time_ms,
              const char** label,
              float* score,
              bool* is_new);

  // Frames actually averaged on the last Update. If this stays below min_count
  // the recognizer is starving and will never report.
  int LastUsedFrames() const { return last_used_; }

  void Reset();

 private:
  static constexpr int kMaxHistory = 12;

  struct Entry {
    int32_t time_ms;
    float   scores[kCategoryCount];
  };

  Entry   history_[kMaxHistory];
  int     count_;
  int     head_;
  int     last_used_;

  int     history_frames_;
  float   detection_thresh_;
  int32_t suppression_ms_;
  int     min_count_;
  int32_t max_age_ms_;

  int         previous_index_;
  int32_t     previous_time_ms_;
  const char* previous_label_;
};
