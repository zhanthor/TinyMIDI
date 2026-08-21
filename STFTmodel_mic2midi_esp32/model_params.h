#pragma once

// ---- audio ---------------------------------------------------------------
constexpr int kSampleRate = 16000;

// 64 ms audio clip.
constexpr int kWindowSamples = 1024;

// ---- STFT ----------------------------------------------------------------
constexpr int kFrameLength = 128;
constexpr int kFrameStep = 32;
constexpr int kNumFrames = 29;
constexpr int kNumFFTBins = 65;

// Total model input elements.
constexpr int kFeatureElements = kNumFrames * kNumFFTBins;

// ---- model ---------------------------------------------------------------
constexpr int kCategoryCount = 28;

constexpr float kRefInputScale = 0.0393452756f;  
constexpr int kRefInputZero = -39;               

constexpr float kRefOutputScale = 0.00390625f;
constexpr int kRefOutputZero = -128;

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