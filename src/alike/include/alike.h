#ifndef ALIKE_ALIKE_H
#define ALIKE_ALIKE_H

#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <iostream>
#include <filesystem>

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include "alike_common.h"

class ALike{
public:
    explicit ALike(const std::string& weights_path, bool use_mask = false);
    ~ALike() = default;

    /// @brief Extract scores map and descriptor map from input the image
    /// @param frame input image
    /// @param frame_mask [optional] mask of regions, which will be zeroed
    void run(const cv::Mat& frame, const cv::Mat& frame_mask);

    float* get_scores_map() { return m_scores_map_tensor->GetTensorMutableData<float>(); }
    float* get_descriptor_map() { return m_descriptor_map_tensor->GetTensorMutableData<float>(); }
    const ALikeMetadata& get_meta() const { return m_meta; }

private:
    /// @brief Preprocess input image and copy it to ORT tensors
    /// @param frame input image
    /// @param frame_mask [optional] mask of regions, which will be zeroed
    void prepare_input_tensor(const cv::Mat& frame, const cv::Mat& frame_mask);

private:
    bool m_use_mask;
    ALikeMetadata m_meta;

    std::vector<const char*> m_input_node_names = {"image"};
    std::vector<const char*> m_output_node_names = {"scores_map", "descriptor_map"}; 

    std::vector<int64_t> m_input_shape;
    std::vector<int64_t> m_scores_map_shape;
    std::vector<int64_t> m_descriptor_map_shape;

    std::vector<float> m_image_mask;
    std::vector<float> m_input_memory;
    std::vector<float> m_scores_map_memory;
    std::vector<float> m_descriptor_map_memory;
    
    std::unique_ptr<Ort::Value> m_input_tensor;
    std::unique_ptr<Ort::Value> m_scores_map_tensor;
    std::unique_ptr<Ort::Value> m_descriptor_map_tensor;
    std::unique_ptr<Ort::Session> m_session;

    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "alike-cuda-environment"};
    Ort::RunOptions m_run_options{nullptr};

    // Not used/does not work
    // std::unique_ptr<Ort::IoBinding> m_io_binding;
    // Ort::AllocatorWithDefaultOptions m_allocator;
};

#endif // ALIKE_ALIKE_H