#include "note_recognizer.h"

#include <string.h>

NoteRecognizer::NoteRecognizer(int history_frames,
                               float detection_thresh,
                               int32_t suppression_ms,
                               int min_count,
                               int32_t max_age_ms) {
  Configure(history_frames, detection_thresh, suppression_ms, min_count,
            max_age_ms);
}

void NoteRecognizer::Configure(int history_frames,
                               float detection_thresh,
                               int32_t suppression_ms,
                               int min_count,
                               int32_t max_age_ms) {
  history_frames_ = history_frames < 1 ? 1
                    : (history_frames > kMaxHistory ? kMaxHistory
                                                    : history_frames);
  detection_thresh_ = detection_thresh;
  suppression_ms_   = suppression_ms;
  min_count_        = min_count < 1 ? 1 : min_count;
  max_age_ms_       = max_age_ms;

  // A min_count larger than the history can never be satisfied. Clamp rather
  // than deadlock silently -- that failure mode is very hard to diagnose from
  // the outside, because the device looks fine and simply says nothing. This is
  // exactly the bug that made the first Nano build report no notes at all.
  if (min_count_ > history_frames_) min_count_ = history_frames_;

  Reset();
}

void NoteRecognizer::Reset() {
  count_ = 0;
  head_  = 0;
  last_used_ = 0;
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

  Entry& e = history_[head_];
  e.time_ms = time_ms;
  memcpy(e.scores, scores, kCategoryCount * sizeof(float));
  head_ = (head_ + 1) % kMaxHistory;
  if (count_ < kMaxHistory) count_++;

  // Average the newest history_frames_ entries, walking backwards from the most
  // recent, skipping anything older than max_age_ms_.
  float averaged[kCategoryCount] = {0.0f};
  int   used = 0;
  for (int back = 0; back < count_ && used < history_frames_; back++) {
    const int idx = (head_ - 1 - back + 2 * kMaxHistory) % kMaxHistory;
    const Entry& h = history_[idx];
    if (time_ms - h.time_ms > max_age_ms_) break;
    for (int c = 0; c < kCategoryCount; c++) averaged[c] += h.scores[c];
    used++;
  }
  last_used_ = used;
  if (used < min_count_) return;

  for (int c = 0; c < kCategoryCount; c++) averaged[c] /= (float)used;

  int   best_index = 0;
  float best_score = averaged[0];
  for (int c = 1; c < kCategoryCount; c++) {
    if (averaged[c] > best_score) { best_score = averaged[c]; best_index = c; }
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

  const bool same = (best_index == previous_index_);
  if (same && (time_ms - previous_time_ms_ < suppression_ms_)) return;

  previous_index_   = best_index;
  previous_time_ms_ = time_ms;
  previous_label_   = kCategoryLabels[best_index];
  *is_new = true;
}
