#include "alike.h"

#include <cstring>
#include "alike_common.h"

ALike::ALike(const std::string&  weights_path, bool use_mask)
    : m_use_mask(use_mask) {
    Ort::SessionOptions session_options;

    // Enable CUDA Execution Provider
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    if (status != nullptr) {
        const char* error_message = Ort::GetApi().GetErrorMessage(status);
        std::cerr << "Error enabling CUDA Execution Provider: " << error_message << std::endl;
        Ort::GetApi().ReleaseStatus(status);
    }

    m_session = std::make_unique<Ort::Session>(m_env, weights_path.c_str(), session_options);

    // Get model input/output dimensions
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input size
    auto input_shape = get_tensor_shape(*m_session, "image", true);
    m_meta.input_rows = input_shape[2]; // (1, 1, X, w)
    m_meta.input_cols = input_shape[3]; // (1, 1, h, X)

    // Get descriptors size
    auto descriptor_map_shape = get_tensor_shape(*m_session, "descriptor_map", false);
    m_meta.channels = descriptor_map_shape[1]; // (1, X, h, w)

    // Set tensor dimensions
    m_input_shape = {1, 3, m_meta.input_rows, m_meta.input_cols};
    m_scores_map_shape = {1, 1, m_meta.input_rows, m_meta.input_cols};
    m_descriptor_map_shape = {1, m_meta.channels, m_meta.input_rows, m_meta.input_cols};

    // Allocate memory for onnxruntime input\output
    if (m_use_mask)
        m_image_mask.resize(m_meta.input_rows * m_meta.input_cols);
    m_input_memory.resize(m_meta.input_rows * m_meta.input_cols * 3);

    // FIXME: these two are not used - I haven't figured out a way to
    // make io_binding work with CUDA, yet. So output tensors are reallocatted each run
    // m_scores_map_memory.resize(m_meta.input_rows * m_meta.input_cols);
    // m_descriptor_map_memory.resize(m_meta.input_rows * m_meta.input_cols * m_meta.channels);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    m_input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        memory_info, 
        m_input_memory.data(), m_input_memory.size(),
        m_input_shape.data(), m_input_shape.size()));


    // FIXME: yet again, these are not used
    // m_scores_map_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
    //     memory_info, 
    //     m_scores_map_memory.data(), m_scores_map_memory.size(),
    //     m_scores_map_shape.data(), m_scores_map_shape.size()));

    // m_descriptor_map_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
    //     memory_info, 
    //     m_descriptor_map_memory.data(), m_descriptor_map_memory.size(),
    //     m_descriptor_map_shape.data(), m_descriptor_map_shape.size()));

    // m_io_binding = std::make_unique<Ort::IoBinding>(*m_session);
    // m_io_binding->BindInput(m_input_node_names[0], *m_input_tensor);
    // m_io_binding->BindOutput(m_output_node_names[0], *m_scores_map_tensor);
    // m_io_binding->BindOutput(m_output_node_names[1], *m_descriptor_map_tensor);
}

void ALike::run(const cv::Mat& frame, const cv::Mat& frame_mask){
    prepare_input_tensor(frame, frame_mask);

    auto outputs = m_session->Run(m_run_options, 
        m_input_node_names.data(), m_input_tensor.get(), m_input_node_names.size(), 
        m_output_node_names.data(), m_output_node_names.size());

    m_scores_map_tensor = std::make_unique<Ort::Value>(std::move(outputs[0]));
    m_descriptor_map_tensor = std::make_unique<Ort::Value>(std::move(outputs[1]));

    // Apply mask to the score map
    if (m_use_mask) {
        int scores_map_size = m_image_mask.size();
        float* scores_map_ptr = get_scores_map();
        for (size_t i = 0; i < scores_map_size; i++){
            scores_map_ptr[i] = scores_map_ptr[i] * m_image_mask[i];
        }
    }
}

void ALike::prepare_input_tensor(const cv::Mat& frame, const cv::Mat& frame_mask){
    auto image_shape = frame.size;
    cv::Mat image(frame.size(), frame.type());
    frame.copyTo(image);

    m_meta.scale_factor = std::fmin(
        static_cast<float>(m_meta.input_rows) / image_shape[0], 
        static_cast<float>(m_meta.input_cols) / image_shape[1]
    );

    // Scale and round to the clossest even number
    int new_rows = static_cast<int>(std::round(image_shape[0] * m_meta.scale_factor / 2) * 2);
    int new_cols = static_cast<int>(std::round(image_shape[1] * m_meta.scale_factor / 2) * 2);

    m_meta.h_padding = (m_meta.input_rows - new_rows) / 2;
    m_meta.v_padding = (m_meta.input_cols - new_cols) / 2;

    cv::resize(image, image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int row_offset = m_meta.h_padding * (new_cols + 2 * m_meta.v_padding);
    int col_offset = m_meta.v_padding * new_rows;
    float* data_ptr = m_input_memory.data();

    // (H, W, C) -> (C, H, W).flatten()
    for (int c = 0; c < 3; c++) {
        data_ptr+=row_offset;
        for (int y = 0; y < new_rows; y++) {
            data_ptr+=col_offset;
            for (int x = 0; x < new_cols; x++) {
                *data_ptr++ = static_cast<float>(image.at<cv::Vec3b>(y, x)[c]) / 255.0f;
            }
            data_ptr+=col_offset;
        }
        data_ptr+=row_offset;
    }

    // Load the mask in the right format
    if (m_use_mask) {
        cv::Mat mask(frame_mask.size(), frame_mask.type());
        frame_mask.copyTo(mask);
        cv::resize(mask, mask, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);

        data_ptr = m_image_mask.data() + row_offset;
        for (int y = 0; y < new_rows; y++) {
            data_ptr+=col_offset;
            for (int x = 0; x < new_cols; x++) {
                *data_ptr++ = mask.at<cv::Vec<uchar, 1>>(y, x)[0] > 127 ? 0.0f : 1.0f; // 'binarify' and invert the mask
            }
            data_ptr+=col_offset;
        }
    }
}