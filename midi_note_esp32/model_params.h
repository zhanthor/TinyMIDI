// AUTO-GENERATED. Do not edit.
#pragma once

// ---- audio front end -----------------------------------------------------
constexpr int   kSampleRate    = 16000;
constexpr int   kWindowSamples = 2048;   // 64 ms
constexpr int   kFFTSize       = 2048;       // zero-padded from kWindowSamples
constexpr int   kNumFFTBins    = 1025;
constexpr int   kNumLogBins    = 217;
constexpr float kLogEps        = 1e-10f;
constexpr float kLogTopRange   = 8.0f;

// Model input is (kNumLogBins, 1, 1).
constexpr int kFeatureElements = 217;

// ---- filterbank geometry (informational) ---------------------------------
constexpr int kMidiLow      = 36;
constexpr int kMidiHigh     = 108;
constexpr int kBinsPerSemi  = 3;

// ---- model ---------------------------------------------------------------
constexpr int kCategoryCount = 28;

constexpr float kRefInputScale  = 0.0369038545f;
constexpr int   kRefInputZero   = -33;
constexpr float kRefOutputScale = 0.0039062500f;
constexpr int   kRefOutputZero  = -128;

inline const char* const kCategoryLabels[kCategoryCount] = {
  "45",
  "46",
  "47",
  "48",
  "49",
  "50",
  "51",
  "52",
  "53",
  "54",
  "55",
  "56",
  "57",
  "58",
  "59",
  "60",
  "61",
  "62",
  "63",
  "64",
  "65",
  "66",
  "67",
  "68",
  "69",
  "70",
  "71",
  "silence"
};

constexpr int kSilenceIndex = 27;
