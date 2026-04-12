#include "fixed_weight_decoder.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app {

// ============================================================
// Training-derived constants (from analysis/decoder_rnn_shay/v9)
// Shay's GRU v9 model — 100ms bins, 15-step window, R²=0.847
// ============================================================

// Channel standard deviations from training (64 channels)
// Used to compute spike thresholds: threshold = sigma * channel_std
static constexpr float kChannelStds[64] = {
  20.298405f, 22.462172f, 21.739756f, 25.051172f, 19.765423f, 20.307899f, 25.570288f, 21.579393f,
  22.996510f, 24.847061f, 23.753981f, 21.673481f, 25.781212f, 24.577623f, 19.569157f, 22.714901f,
  22.350138f, 25.650391f, 25.125837f, 20.648691f, 22.774551f, 25.506161f, 26.347424f, 25.485119f,
  21.127657f, 22.593182f, 20.331587f, 23.872692f, 24.270782f, 22.835041f, 21.272261f, 20.669004f,
  23.986588f, 25.591860f, 23.556793f, 19.685074f, 20.843586f, 24.842119f, 25.459375f, 24.482803f,
  23.969755f, 23.472115f, 19.928993f, 24.721952f, 25.802179f, 19.781519f, 23.407574f, 24.860622f,
  19.330820f, 22.719198f, 22.345249f, 24.027863f, 22.243345f, 24.570818f, 26.009296f, 24.696100f,
  21.563696f, 23.315105f, 23.492218f, 24.907070f, 22.085670f, 21.834814f, 25.990767f, 20.744581f,
};

// Feature means from training (192 features = 64 channels x 3 thresholds)
static constexpr float kFeatMean[192] = {
  15.685611f, 26.608480f, 23.329424f, 38.542477f, 13.320252f, 15.976094f, 40.938656f, 22.171854f,
  29.305819f, 37.691025f, 32.667419f, 22.849045f, 41.961060f, 36.480377f, 12.381597f, 27.800482f,
  26.202677f, 41.447754f, 38.862125f, 17.458878f, 27.990828f, 40.774620f, 43.872803f, 40.562172f,
  20.138475f, 27.400541f, 16.025410f, 33.355286f, 35.358894f, 28.532251f, 20.884228f, 17.721996f,
  34.021049f, 41.141483f, 31.762291f, 12.846790f, 18.764997f, 37.829197f, 40.622913f, 36.219517f,
  33.918961f, 31.355434f, 14.088408f, 37.284317f, 41.995789f, 13.466997f, 31.181326f, 37.839874f,
  11.202676f, 28.046459f, 26.009022f, 33.990829f, 25.752970f, 36.578861f, 42.781837f, 37.159676f,
  21.963163f, 30.865284f, 31.672981f, 37.853405f, 24.871599f, 23.775221f, 42.706360f, 18.063900f,
  10.333784f, 18.261013f, 16.025410f, 25.402346f, 8.321606f, 10.631333f, 26.778229f, 15.274395f,
  19.987070f, 24.838671f, 22.005262f, 15.703653f, 27.309877f, 24.193956f, 7.655089f, 19.005413f,
  18.009022f, 27.046761f, 25.607277f, 11.797925f, 19.113216f, 26.684710f, 28.490904f, 26.562773f,
  13.777778f, 18.741993f, 10.712073f, 22.386257f, 23.585627f, 19.481131f, 14.301609f, 11.987370f,
  22.749962f, 26.841827f, 21.420538f, 8.015036f, 12.807398f, 24.988123f, 26.555405f, 24.063900f,
  22.741241f, 21.225531f, 9.122538f, 24.690271f, 27.352428f, 8.553601f, 21.112764f, 24.950384f,
  6.603669f, 19.179371f, 17.844835f, 22.724703f, 17.584423f, 24.263119f, 27.791309f, 24.568186f,
  15.038190f, 20.915953f, 21.390467f, 25.016388f, 17.058939f, 16.362202f, 27.782740f, 12.276650f,
  7.627424f, 13.115622f, 11.476019f, 18.956697f, 6.233499f, 7.787400f, 20.307472f, 10.983461f,
  14.400241f, 18.543528f, 16.044655f, 11.298451f, 20.755676f, 17.928581f, 5.738986f, 13.694933f,
  12.894301f, 20.560968f, 19.192753f, 8.598256f, 13.786198f, 20.163584f, 22.047060f, 20.092617f,
  9.958954f, 13.464441f, 7.835664f, 16.297850f, 17.285521f, 13.989926f, 10.319050f, 8.698091f,
  16.622913f, 20.367764f, 15.577207f, 5.996241f, 9.269584f, 18.651781f, 20.094271f, 17.811007f,
  16.604271f, 15.389265f, 6.751165f, 18.341602f, 20.852955f, 6.362652f, 15.285069f, 18.597204f,
  5.010976f, 13.757480f, 12.826643f, 16.648624f, 12.662758f, 18.003910f, 21.298151f, 18.245678f,
  10.864983f, 15.134416f, 15.488197f, 18.627424f, 12.267930f, 11.708916f, 21.293640f, 8.896256f,
};

// Feature stds from training (192 features)
static constexpr float kFeatStd[192] = {
  18.191763f, 20.244968f, 19.546118f, 21.780008f, 16.925415f, 17.365368f, 21.177618f, 20.224771f,
  20.438295f, 21.930269f, 21.179695f, 19.591692f, 21.828548f, 22.969292f, 15.422401f, 21.149147f,
  20.286938f, 21.253248f, 21.969234f, 18.089228f, 22.952213f, 21.771219f, 21.457684f, 21.828251f,
  19.031542f, 21.259468f, 18.069740f, 21.133158f, 21.509619f, 20.265467f, 19.803974f, 17.809914f,
  22.968887f, 21.598808f, 22.355989f, 15.909631f, 18.886637f, 21.482899f, 21.468498f, 22.618111f,
  21.605556f, 21.928602f, 16.836626f, 21.929855f, 22.115938f, 16.282793f, 21.008196f, 22.103706f,
  14.593416f, 20.620222f, 20.135286f, 23.551311f, 20.698746f, 22.340530f, 21.584255f, 21.792974f,
  21.297247f, 20.777351f, 21.789530f, 21.658014f, 21.408495f, 19.772276f, 21.563169f, 18.470993f,
  13.594925f, 14.713108f, 14.304099f, 15.001314f, 12.793591f, 13.004782f, 14.492745f, 14.749949f,
  14.694764f, 15.094151f, 14.973563f, 14.382947f, 14.873932f, 15.829131f, 11.840421f, 15.128487f,
  14.708593f, 14.621587f, 15.133408f, 13.515504f, 16.285362f, 14.937003f, 14.659839f, 14.974978f,
  14.044764f, 15.235609f, 13.657270f, 14.922446f, 15.108807f, 14.564578f, 14.461169f, 13.288880f,
  15.942496f, 14.819059f, 15.689116f, 12.088410f, 14.043472f, 14.840352f, 14.662546f, 15.678202f,
  15.155654f, 15.444760f, 12.784846f, 15.212642f, 15.066474f, 12.426658f, 14.840106f, 15.266517f,
  11.372702f, 14.854069f, 14.563771f, 16.258026f, 14.905855f, 15.443290f, 14.699476f, 15.081415f,
  15.459086f, 14.839217f, 15.339760f, 14.988008f, 15.440960f, 14.430024f, 14.668025f, 13.757895f,
  10.661610f, 11.118814f, 10.920627f, 11.549815f, 10.213261f, 10.217992f, 11.281211f, 11.224424f,
  11.116843f, 11.696676f, 11.402411f, 10.994255f, 11.582550f, 12.098734f, 9.567594f, 11.437461f,
  11.145898f, 11.447211f, 11.649822f, 10.488624f, 12.257290f, 11.604843f, 11.615788f, 11.647099f,
  10.821732f, 11.484796f, 10.675017f, 11.359497f, 11.491159f, 11.042858f, 11.027085f, 10.331665f,
  12.058995f, 11.557966f, 11.918956f, 9.719776f, 10.827259f, 11.486478f, 11.430188f, 12.000022f,
  11.520167f, 11.747769f, 10.183382f, 11.654526f, 11.766397f, 9.952143f, 11.285550f, 11.715734f,
  9.356391f, 11.189957f, 11.071066f, 12.333899f, 11.262832f, 11.791668f, 11.475060f, 11.662204f,
  11.740227f, 11.251074f, 11.594770f, 11.589977f, 11.705737f, 10.932086f, 11.461974f, 10.619514f,
};

// ============================================================
// Implementation
// ============================================================

template <typename T>
T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

FixedWeightDecoder::FixedWeightDecoder() : publish_rate_limiter_(kPublishRateSec) {
  // Copy training constants into member arrays
  for (size_t i = 0; i < kNumNeural; ++i) {
    channel_stds_[i] = kChannelStds[i];
  }
  for (size_t i = 0; i < kNumFeatures; ++i) {
    feat_mean_[i] = kFeatMean[i];
    feat_std_[i] = kFeatStd[i];
  }
}

void FixedWeightDecoder::initialize_thresholds() {
  // Compute per-channel thresholds: threshold[ch][t] = sigma[t] * channel_std[ch]
  for (size_t ch = 0; ch < kNumNeural; ++ch) {
    for (size_t t = 0; t < kNumThresholds; ++t) {
      channel_thresholds_[ch][t] = kSigmaThresholds[t] * channel_stds_[ch];
    }
  }
  spdlog::info("Initialized 3-threshold spike detection (3σ/4σ/5σ) for {} channels", kNumNeural);
}

bool FixedWeightDecoder::setup() {
  if (!get_app_config(
          [this](const synapse::ApplicationNodeConfig& configuration) {
            return validate_config(configuration);
          },
          application_config_)) {
    spdlog::error("Failed to get app config");
    return false;
  }

  if (!parse_config(application_config_)) {
    spdlog::error("Failed to parse app config");
    return false;
  }

  // Initialize spike thresholds from training channel stds
  initialize_thresholds();

  const uint32_t broadband_node_id = 1;
  if (!setup_reader(broadband_node_id)) {
    spdlog::warn("Failed to set up reader for controller");
    return false;
  }

  // Output tap: 7 values [joy_x, joy_y, rot, depth, lt, rt, gate]
  if (!create_tap<synapse::Tensor>("joystick_out")) {
    spdlog::warn("Failed to create tap for joystick out");
    return false;
  }

  if (enable_function_profiling_) {
    function_profiler_manager_.add("full_loop");
    function_profiler_manager_.add("inference");
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
  }

  if (enable_inference_) {
    setup_inference();
  }

  return true;
}

void FixedWeightDecoder::main() {
  const float bin_size_ms = 100;
  std::vector<synapse::BroadbandFrame> broadband_frames;

  while (node_running_) {
    if (!wait_for_frames(broadband_frames, bin_size_ms)) {
      continue;
    }

    start_profile("full_loop");

    // Initialize filters on first frame
    const auto broadband_frame = broadband_frames.at(0);
    if (!filters_initialized_) {
      const size_t channel_count = broadband_frame.frame_data_size();
      const float sample_rate_hz = broadband_frame.sample_rate_hz();
      sample_rate_hz_ = sample_rate_hz;

      spdlog::info("Received first frames: {} channels at {} Hz", channel_count, sample_rate_hz);

      if (channel_count < kNumNeural) {
        spdlog::error("Expected at least {} neural channels, got {}", kNumNeural, channel_count);
        return;
      }

      initialize_filters(channel_count, sample_rate_hz, bin_size_ms);
      continue;
    }

    // Bandpass filter all channels for this bin
    const size_t n_channels = broadband_frames.at(0).frame_data_size();
    std::vector<std::vector<float>> filtered_channel_data(n_channels);
    for (auto& ch_data : filtered_channel_data) {
      ch_data.reserve(broadband_frames.size());
    }

    for (const auto& frame : broadband_frames) {
      const auto& frame_data = frame.frame_data();
      for (int ch = 0; ch < frame_data.size(); ++ch) {
        auto& filter = bandpass_filters_.at(ch);
        float filtered = filter->filter(frame_data[ch]);
        filtered_channel_data.at(ch).push_back(filtered);
      }
    }

    // Extract 192 features (3-threshold spike counts) from this bin
    auto features = extract_features(filtered_channel_data);

    // Normalize using training statistics
    normalize_features(features);

    // Add to circular buffer
    feature_buffer_.push_back(features);
    if (feature_buffer_.size() > kSeqLen) {
      feature_buffer_.pop_front();
    }

    // Default outputs: all zeros
    std::array<float, kNumOutputs> outputs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Run inference once we have a full window
    if (feature_buffer_.size() == kSeqLen && enable_inference_ && model_ && model_->is_ready()) {
      outputs = run_inference();
    }

    // Apply gate: if gate < threshold, suppress joystick axes
    float gate = outputs[6];
    if (gate < kGateThreshold) {
      outputs[0] = 0.0f;  // joy_x
      outputs[1] = 0.0f;  // joy_y
      outputs[2] = 0.0f;  // rot
      outputs[3] = 0.0f;  // depth
    }

    // Publish output tensor: 7 values
    synapse::Tensor output_tensor;
    const auto tensor_shape = {static_cast<int>(kNumOutputs)};
    output_tensor.mutable_shape()->Add(tensor_shape.begin(), tensor_shape.end());
    output_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    output_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

    std::vector<float> output_vec(outputs.begin(), outputs.end());
    output_tensor.set_data(synapse::pack_tensor_data(output_vec));

    const auto current_time_ns = synapse::get_steady_clock_now();
    output_tensor.set_timestamp_ns(current_time_ns.count());

    if (publish_rate_limiter_.reset_if_elapsed()) {
      if (publish_tap("joystick_out", output_tensor)) {
        spdlog::info("Published: joy=[{:.3f},{:.3f}] rot={:.3f} depth={:.3f} trig=[{:.3f},{:.3f}] gate={:.3f}",
                     outputs[0], outputs[1], outputs[2], outputs[3],
                     outputs[4], outputs[5], outputs[6]);
      } else {
        spdlog::warn("Failed to publish tensor data");
      }
      stop_profile("full_loop");
      print_profile("full_loop");
    }
  }
}

std::array<float, kNumFeatures> FixedWeightDecoder::extract_features(
    const std::vector<std::vector<float>>& filtered_channel_data) {
  std::array<float, kNumFeatures> features = {};

  // For each neural channel, count threshold crossings at 3σ, 4σ, 5σ
  for (size_t ch = 0; ch < kNumNeural && ch < filtered_channel_data.size(); ++ch) {
    const auto& ch_data = filtered_channel_data[ch];

    for (size_t t = 0; t < kNumThresholds; ++t) {
      float threshold = channel_thresholds_[ch][t];
      uint32_t count = 0;

      for (const float sample : ch_data) {
        if (std::fabs(sample) > threshold) {
          count++;
        }
      }

      // Feature layout: [ch0_3σ, ch1_3σ, ..., ch63_3σ, ch0_4σ, ..., ch63_4σ, ch0_5σ, ..., ch63_5σ]
      features[t * kNumNeural + ch] = static_cast<float>(count);
    }
  }

  return features;
}

void FixedWeightDecoder::normalize_features(std::array<float, kNumFeatures>& features) {
  for (size_t i = 0; i < kNumFeatures; ++i) {
    features[i] = (features[i] - feat_mean_[i]) / (feat_std_[i] + 1e-8f);
  }
}

std::array<float, kNumOutputs> FixedWeightDecoder::run_inference() {
  std::array<float, kNumOutputs> outputs = {};

  auto inputs = model_->get_input_info();
  if (inputs.empty()) {
    return outputs;
  }

  // Build input tensor: shape (1, 192, 30) — features × time
  // Layout: for each feature f, then for each time step t
  std::vector<float> input_data(kNumFeatures * kSeqLen, 0.0f);

  for (size_t f = 0; f < kNumFeatures; ++f) {
    for (size_t t = 0; t < kSeqLen; ++t) {
      // input_data[f * kSeqLen + t] = feature_buffer_[t][f]
      input_data[f * kSeqLen + t] = feature_buffer_[t][f];
    }
  }

  start_profile("inference");
  auto result = model_->infer({input_data});
  stop_profile("inference");
  print_profile("inference");

  if (!result.success || result.outputs.empty()) {
    spdlog::warn("Inference failed");
    return outputs;
  }

  // Update benchmarking
  inference_count_++;
  inference_total_us_ += result.inference_time_us;
  inference_min_us_ = std::min(inference_min_us_, result.inference_time_us);
  inference_max_us_ = std::max(inference_max_us_, result.inference_time_us);

  if (inference_count_ % 100 == 0) {
    uint64_t avg_us = inference_total_us_ / inference_count_;
    spdlog::info("Inference stats: count={}, avg={} us, min={} us, max={} us",
                  inference_count_, avg_us, inference_min_us_, inference_max_us_);
  }

  // Parse 7 outputs: [joy_x, joy_y, rot, depth, lt, rt, gate]
  const auto& out = result.outputs[0];
  for (size_t i = 0; i < kNumOutputs && i < out.size(); ++i) {
    outputs[i] = out[i];
  }

  // Clamp joystick axes to [-1, 1]
  for (size_t i = 0; i < 4; ++i) {
    outputs[i] = clamp(outputs[i], -1.0f, 1.0f);
  }
  // Clamp triggers and gate to [0, 1]
  for (size_t i = 4; i < 7; ++i) {
    outputs[i] = clamp(outputs[i], 0.0f, 1.0f);
  }

  return outputs;
}

void FixedWeightDecoder::setup_inference() {
  auto runtimes = synapse::get_available_runtimes();
  spdlog::info("Available inference runtimes:");
  for (const auto& rt : runtimes) {
    const char* name = "unknown";
    switch (rt) {
      case synapse::InferenceRuntime::kCpu: name = "CPU (ONNX Runtime)"; break;
      case synapse::InferenceRuntime::kGpu: name = "GPU (QNN)"; break;
      case synapse::InferenceRuntime::kDsp: name = "DSP (QNN HTP)"; break;
      case synapse::InferenceRuntime::kAuto: name = "Auto"; break;
    }
    spdlog::info("  - {}", name);
  }

  model_ = synapse::create_model(model_name_);

  if (model_ && model_->is_ready()) {
    spdlog::info("Model '{}' loaded — input: (1, {}, {}), output: (1, {})",
                 model_name_, kNumFeatures, kSeqLen, kNumOutputs);

    auto inputs = model_->get_input_info();
    for (const auto& input : inputs) {
      std::string shape_str;
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(input.shape[i]);
      }
      spdlog::info("  Input: {} shape=[{}] elements={}", input.name, shape_str, input.element_count);
    }

    auto model_outputs = model_->get_output_info();
    for (const auto& output : model_outputs) {
      std::string shape_str;
      for (size_t i = 0; i < output.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(output.shape[i]);
      }
      spdlog::info("  Output: {} shape=[{}] elements={}", output.name, shape_str, output.element_count);
    }
  } else {
    spdlog::warn("Model '{}' not available — deploy with: synapsectl deploy-model decoder.onnx --name {} -u <device>",
                  model_name_, model_name_);
  }
}

bool FixedWeightDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                         float bin_size_ms) {
  if (bin_size_ms <= 0) {
    spdlog::warn("invalid bin size of: {}", bin_size_ms);
    return false;
  }

  const float bin_size_sec = bin_size_ms / 1000;
  const size_t target_num_of_frames = bin_size_sec * sample_rate_hz_;

  frames.clear();

  while (node_running_) {
    auto messages = data_reader_->receive_multipart();
    if (messages.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    frames.reserve(frames.size() + messages.size());

    for (auto& message : messages) {
      const auto maybe_frame =
          synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value()) {
        spdlog::warn("Failed to parse broadband frame");
        if (frames.empty()) {
          return false;
        }
        return true;
      }

      const auto& broadband_frame = maybe_frame.value();

      const auto dropped_frames =
          detect_dropped_frames(last_sequence_number_, broadband_frame.sequence_number());
      if (dropped_frames != 0) {
        spdlog::warn("Dropped: {} frames", dropped_frames);
      }
      last_sequence_number_ = broadband_frame.sequence_number();

      frames.push_back(broadband_frame);
    }

    if (frames.size() >= target_num_of_frames) {
      return true;
    }
  }
  return false;
}

int FixedWeightDecoder::detect_dropped_frames(const uint64_t last_sequence_number,
                                              const uint64_t current_sequence_number) {
  const auto expected_sequence_number = last_sequence_number + 1;
  return (current_sequence_number - expected_sequence_number);
}

void FixedWeightDecoder::initialize_filters(const size_t channel_count, const float sample_rate_hz,
                                            const float bin_size_ms) {
  spdlog::info("Initializing: sample_rate={} Hz, channels={}, bin_size={} ms",
               sample_rate_hz, channel_count, bin_size_ms);

  bandpass_filters_.clear();
  bandpass_filters_.reserve(channel_count);
  for (size_t ch = 0; ch < channel_count; ++ch) {
    auto filter_ptr = synapse::create_bandpass_filter<kSpectralFilterOrder>(
        sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
    if (filter_ptr == nullptr) {
      spdlog::error("Failed to create filter for channel: {}", ch);
    }
    bandpass_filters_.push_back(std::move(filter_ptr));
  }
  spdlog::info("Initialized {} bandpass filters ({}-{} Hz)", channel_count, low_cutoff_hz_, high_cutoff_hz_);
  filters_initialized_ = true;
}

bool FixedWeightDecoder::validate_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();

  const std::vector<std::string> required = {
    "low_cutoff_hz", "high_cutoff_hz", "enable_function_profiling"
  };

  for (const auto& key : required) {
    if (!parameters.contains(key)) {
      spdlog::error("{} not found in configuration", key);
      return false;
    }
  }

  return true;
}

bool FixedWeightDecoder::parse_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();
  try {
    low_cutoff_hz_ = parameters.at("low_cutoff_hz").number_value();
    high_cutoff_hz_ = parameters.at("high_cutoff_hz").number_value();
    enable_function_profiling_ = parameters.at("enable_function_profiling").bool_value();

    if (parameters.contains("enable_inference")) {
      enable_inference_ = parameters.at("enable_inference").bool_value();
    }
    if (parameters.contains("model_name")) {
      model_name_ = parameters.at("model_name").string_value();
    }

    spdlog::info("Config: filter={}-{} Hz, inference={}, model={}",
                 low_cutoff_hz_, high_cutoff_hz_, enable_inference_, model_name_);
    application_config_ = configuration;
    return true;
  } catch (const std::exception& e) {
    spdlog::error("Failed to parse configuration: {}", e.what());
    return false;
  }
}

}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
