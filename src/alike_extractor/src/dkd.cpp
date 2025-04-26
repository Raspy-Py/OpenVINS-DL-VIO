#include "dkd.h"

#include <cmath>
#include <iostream>

Eigen::MatrixXi DKD::tiled_detect_keypoints(const float* scores_map, int h, int w, int ph, int pv) {
    int num_tiles_h = h / m_radius;
    int num_tiles_w = w / m_radius;

    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> keypoints(m_top_k, 2);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> scores(scores_map, h, w);

    std::vector<std::pair<float, int>> all_keypoints;
    all_keypoints.reserve(num_tiles_h * num_tiles_w);

    int i_start = static_cast<int>(std::ceil(static_cast<float>(ph + m_padding) / static_cast<float>(m_radius))) * m_radius;
    int j_start = static_cast<int>(std::ceil(static_cast<float>(pv + m_padding) / static_cast<float>(m_radius))) * m_radius;
    int i_stop = num_tiles_h * m_radius - i_start;
    int j_stop = num_tiles_w * m_radius - j_start;

    for (int start_row = i_start; start_row < i_stop; start_row+=m_radius) {
        for (int start_col = j_start; start_col < j_stop; start_col+=m_radius) {            
            float max_val = -std::numeric_limits<float>::max();
            int max_idx = -1;

            // FIXME: with Eigen::block and Eigen::maxCoeff
            for (int ki = 0; ki < m_radius; ++ki) {
                for (int kj = 0; kj < m_radius; ++kj) {
                    float val = scores(start_row + ki, start_col + kj);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (start_row + ki) * w + (start_col + kj);
                    }
                }
            }
            all_keypoints.emplace_back(max_val, max_idx);
        }
    }

    std::partial_sort(all_keypoints.begin(), 
                      all_keypoints.begin() + std::min(m_top_k, static_cast<int>(all_keypoints.size())), 
                      all_keypoints.end(),
                      std::greater<std::pair<float, int>>());

    int actual_top_k = std::min(m_top_k, static_cast<int>(all_keypoints.size()));
    keypoints.conservativeResize(actual_top_k, Eigen::NoChange);

    for (int i = 0; i < actual_top_k; ++i) {
        const auto& [val, idx] = all_keypoints[i];
        keypoints(i, 1) = idx / w;
        keypoints(i, 0) = idx % w;
    }

    return keypoints;
}

Eigen::MatrixXi DKD::maxpool_detect_keypoints(const float* scores_map, int h, int w, int ph, int pv){
    // Map existing scores data the the eigen matrix
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> scores(scores_map, h, w);
    
    int h_start = ph + m_padding;
    int w_start = pv + m_padding;
    int h_stop = h - h_start;
    int w_stop = w - w_start;

    auto maxpooled = maxpool2d(
        scores,
        h_start, w_start,
        h_stop, w_stop, 
        m_radius
    );   

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> max_mask = (scores.array() == maxpooled.array()).matrix();
    
    auto cmp = [&scores](const Eigen::Vector2i& a, const Eigen::Vector2i& b) {
        return scores(a.y(), a.x()) > scores(b.y(), b.x());
    };
    std::priority_queue<Eigen::Vector2i, std::vector<Eigen::Vector2i>, decltype(cmp)> pq(cmp);
    
    // Extarct keypoints from the score map using maximum mask
    for (int i = h_start; i < h_stop; ++i) {
        for (int j = w_start; j < w_stop; ++j) {
            if (max_mask(i, j)) {
                pq.push(Eigen::Vector2i(j, i));
                if (pq.size() > m_top_k) {
                    pq.pop();
                }
            }
        }
    }
    
    // Select top k keypoints by their value in the score map
    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> keypoints(m_top_k, 2);
    for (int i = m_top_k - 1; i >= 0; --i) {
        if (!pq.empty()) {
            keypoints.row(i) = pq.top();
            pq.pop();
        }
    }
    
    return keypoints;
}

Eigen::MatrixXf DKD::sample_descriptors(const float* descriptor_map, const Eigen::MatrixXi& kpts, int c, int h, int w) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> descriptors_reshaped(descriptor_map, c, h * w);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> descriptors(kpts.rows(), c);

    for (int i = 0; i < kpts.rows(); ++i) {
        int x = kpts(i, 0);
        int y = kpts(i, 1);
        
        // FIXME: remove in release, cause out-of-bounds should never hapen if the code is right
        // Add boundary checks
        if (x >= 0 && x < w && y >= 0 && y < h) {
            descriptors.row(i) = descriptors_reshaped.col(y * w + x).transpose();
        } else {
            std::cout << "Out of bound detected: " << x << ", " << y << std::endl;
            // Handle out-of-bounds access, e.g., set to zero or some default value
            descriptors.row(i).setZero();
        }
    }

    descriptors.rowwise().normalize();

    return descriptors;
}

void DKD::run(const float* scores_map, const float* descriptor_map, const ALikeMetadata& meta, 
              Eigen::MatrixXi& keypoints, Eigen::MatrixXf& descriptors){
    int c = meta.channels, h = meta.input_rows, w = meta.input_cols, ph = meta.h_padding, pv = meta.v_padding;

    // Extract keypoints' UVs
    // Two detection methods can be used interchangably. 
    // Yet, maxpool currently yeilds better results
    // keypoints = tiled_detect_keypoints(scores_map, h, w, ph, pv);
    keypoints = maxpool_detect_keypoints(scores_map, h, w, ph, pv);

    // Sample descriptors using UVs
    descriptors = sample_descriptors(descriptor_map, keypoints, c, h, w);

    // Remove padding, added by ALike preprocesing, from keypoints
    keypoints.col(0).array() -= pv; // left padding
    keypoints.col(1).array() -= ph; // top padding

    // Rescale keypoint UVs to the original image resolution
    if (meta.scale_factor > 0.0f && meta.scale_factor != 1.0f){
        float inv_scale_factor = 1.0f / meta.scale_factor;
        keypoints = (keypoints.cast<float>() * inv_scale_factor).array().round().cast<int>();
    }
}


/*
* Helper methods
*/

Eigen::MatrixXf DKD::maxpool2d(const Eigen::MatrixXf& input, int h_start, int w_start, int h_stop, int w_stop, int radius, int stride) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output = Eigen::MatrixXf::Zero(input.rows(), input.cols());
    int kernel_size = 2 * radius + 1;

    for (int i = h_start; i < h_stop; i += stride) {
        for (int j = w_start; j < w_stop; j += stride) {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> kernel = input.block(i - radius, j - radius, kernel_size, kernel_size);
            output(i, j) = kernel.maxCoeff();
        }
    }

    return output;
}