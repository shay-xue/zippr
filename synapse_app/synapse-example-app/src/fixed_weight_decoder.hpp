#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>
#include <synapse-app-sdk/inference/model.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"

#include <google/protobuf/struct.pb.h>

namespace app {

// --- Model constants ---
constexpr size_t kNumNeural = 64;         // Neural channels (0-63)
constexpr size_t kNumThresholds = 3;      // 3σ, 4σ, 5σ
constexpr size_t kNumFeatures = kNumNeural * kNumThresholds;  // 192
constexpr size_t kSeqLen = 15;            // 15 bins × 100ms = 1500ms
constexpr size_t kNumOutputs = 7;         // joy_x, joy_y, rot, depth, lt, rt, gate
constexpr float kSigmaThresholds[kNumThresholds] = {3.0f, 4.0f, 5.0f};
constexpr float kGateThreshold = 0.5f;

// 10 Hz publish rate
constexpr auto kPublishRateSec = 1.0 / 10.0;

class FixedWeightDecoder : public synapse::App {
 public:
  FixedWeightDecoder();

  virtual bool setup() override;

 protected:
  virtual void main() override;

 private:
  synapse::ApplicationNodeConfig application_config_;

  uint64_t last_sequence_number_ = 0;

  synapse::Timer publish_rate_limiter_;

  // Bandpass filters (200-5000 Hz) per channel
  std::atomic<bool> filters_initialized_{false};
  float low_cutoff_hz_ = 200.0f;
  float high_cutoff_hz_ = 5000.0f;
  static constexpr int kSpectralFilterOrder = 2;
  std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

  float sample_rate_hz_ = 32000.0f;

  // Channel standard deviations from training (for threshold computation)
  std::array<float, kNumNeural> channel_stds_;

  // Per-channel thresholds: channel_thresholds_[ch][thresh_idx]
  std::array<std::array<float, kNumThresholds>, kNumNeural> channel_thresholds_;

  // Feature normalization from training
  std::array<float, kNumFeatures> feat_mean_;
  std::array<float, kNumFeatures> feat_std_;

  // Circular buffer: last kSeqLen bins of 192 features each
  std::deque<std::array<float, kNumFeatures>> feature_buffer_;

  // Profiling
  bool enable_function_profiling_ = false;

  // Inference model
  bool enable_inference_ = false;
  std::string model_name_ = "decoder";
  std::unique_ptr<synapse::BaseModel> model_;

  // Inference benchmarking
  uint64_t inference_count_ = 0;
  uint64_t inference_total_us_ = 0;
  uint64_t inference_min_us_ = UINT64_MAX;
  uint64_t inference_max_us_ = 0;

  void setup_inference();
  void initialize_thresholds();

  // Extract 192 features (3-threshold spike counts) from one bin of filtered data
  std::array<float, kNumFeatures> extract_features(
      const std::vector<std::vector<float>>& filtered_channel_data);

  // Normalize features using training mean/std
  void normalize_features(std::array<float, kNumFeatures>& features);

  // Build model input from feature buffer and run inference
  // Returns 7 outputs: [joy_x, joy_y, rot, depth, lt, rt, gate]
  std::array<float, kNumOutputs> run_inference();

  bool wait_for_frames(std::vector<synapse::BroadbandFrame>& frames, const float bin_size_ms);
  int detect_dropped_frames(const uint64_t last, const uint64_t current);
  void initialize_filters(const size_t channel_count, const float sample_rate_hz,
                          const float bin_size_ms);
  bool validate_config(const synapse::ApplicationNodeConfig& configuration);
  bool parse_config(const synapse::ApplicationNodeConfig& configuration);
};
}  // namespace app
