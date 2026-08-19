// note_recognizer.h -- posterior smoothing over a sliding window of results.
//
// Functionally this is Lab 5's recognize_commands.{h,cpp}, reimplemented for
// 33 classes and float scores. The idea is identical: a single inference is
// noisy, so average the class scores over a short history, require the winner
// to clear a confidence threshold, and suppress repeats until either the note
// changes or a refractory period expires.
//
// Without this you get a torrent of flickering output, because at a 32 ms hop
// you are running ~31 inferences per second and adjacent semitones will trade
// places constantly on note onsets and decays.

#pragma once

#include <stdint.h>

#include "model_params.h"

class NoteRecognizer {
 public:
  // history_ms      : averaging span. 4-6 inferences is a good starting point.
  // detection_thresh: minimum averaged score (0-1) for the winner.
  // suppression_ms  : minimum gap before the SAME note reports again.
  // min_count       : minimum inferences in the window before reporting.
  NoteRecognizer(int32_t history_ms = 160,
                 float detection_thresh = 0.55f,
                 int32_t suppression_ms = 250,
                 int min_count = 3);

  // Feeds one dequantized score vector (length kCategoryCount, summing to ~1).
  // Sets *label / *score to the smoothed winner and *is_new to true only on a
  // fresh detection worth printing.
  void Update(const float* scores,
              int32_t time_ms,
              const char** label,
              float* score,
              bool* is_new);

  void Reset();

 private:
  static constexpr int kMaxHistory = 24;

  struct Entry {
    int32_t time_ms;
    float   scores[kCategoryCount];
  };

  Entry   history_[kMaxHistory];
  int     count_;
  int     head_;

  int32_t history_ms_;
  float   detection_thresh_;
  int32_t suppression_ms_;
  int     min_count_;

  int         previous_index_;
  int32_t     previous_time_ms_;
  const char* previous_label_;
};
