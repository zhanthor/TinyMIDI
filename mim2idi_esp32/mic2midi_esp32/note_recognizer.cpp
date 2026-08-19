#include "note_recognizer.h"

#include <string.h>

NoteRecognizer::NoteRecognizer(int32_t history_ms,
                               float detection_thresh,
                               int32_t suppression_ms,
                               int min_count)
    : history_ms_(history_ms),
      detection_thresh_(detection_thresh),
      suppression_ms_(suppression_ms),
      min_count_(min_count) {
  Reset();
}

void NoteRecognizer::Reset() {
  count_ = 0;
  head_  = 0;
  previous_index_   = -1;
  previous_time_ms_ = -1000000;
  previous_label_   = "silence";
}

void NoteRecognizer::Update(const float* scores,
                            int32_t time_ms,
                            const char** label,
                            float* score,
                            bool* is_new) {
  *label  = previous_label_;
  *score  = 0.0f;
  *is_new = false;

  // Push into the ring buffer.
  Entry& e = history_[head_];
  e.time_ms = time_ms;
  memcpy(e.scores, scores, kCategoryCount * sizeof(float));
  head_ = (head_ + 1) % kMaxHistory;
  if (count_ < kMaxHistory) count_++;

  // Average every entry inside the history window.
  float averaged[kCategoryCount] = {0.0f};
  int   used = 0;
  for (int i = 0; i < count_; i++) {
    const Entry& h = history_[i];
    if (time_ms - h.time_ms > history_ms_) continue;
    for (int c = 0; c < kCategoryCount; c++) averaged[c] += h.scores[c];
    used++;
  }
  if (used < min_count_) return;

  for (int c = 0; c < kCategoryCount; c++) averaged[c] /= (float)used;

  int   best_index = 0;
  float best_score = averaged[0];
  for (int c = 1; c < kCategoryCount; c++) {
    if (averaged[c] > best_score) {
      best_score = averaged[c];
      best_index = c;
    }
  }

  *score = best_score;
  *label = kCategoryLabels[best_index];

  if (best_score < detection_thresh_) return;

  // Silence is a state, not an event -- track it but never announce it.
  if (best_index == kSilenceIndex) {
    previous_index_ = best_index;
    previous_label_ = kCategoryLabels[best_index];
    return;
  }

  const bool same_as_before = (best_index == previous_index_);
  const bool still_suppressed =
      same_as_before && (time_ms - previous_time_ms_ < suppression_ms_);
  if (still_suppressed) return;

  previous_index_   = best_index;
  previous_time_ms_ = time_ms;
  previous_label_   = kCategoryLabels[best_index];
  *is_new = true;
}
