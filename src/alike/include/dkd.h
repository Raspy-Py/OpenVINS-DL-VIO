#ifndef ALIKE_DKD_H
#define ALIKE_DKD_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#include "alike_common.h"

class DKD {
public:
    DKD(int top_k = 500, int radius = 4, int padding = 2)
        : m_top_k(top_k), m_radius(radius), m_padding(padding) {}
    
    void run(const float* scores_map, const float* descriptor_map, const ALikeMetadata& meta, 
             Eigen::MatrixXi& keypoints, Eigen::MatrixXf& descriptors);

private:
    // From a pretty random number of detected keypoints only top k
    // with the most intensity will be selected
    int m_top_k;

    // Radius of the local region in which we seek for the maximum intensity
    // points on the score map
    int m_radius;

    // ALike tends to give false postives close to borders. Thus we skip
    // several rows in the score map during keypoint detection
    int m_padding;
    
private:
    /// @brief Extract keypoints with highest local scores from the the score map
    /// @return Matrix (num_pts x 2) of extracted keypoints
    Eigen::MatrixXi tiled_detect_keypoints(const float* scores_map, int h, int w, int ph, int pv);

    /// @brief Extract keypoints from the the score map using simple nms via single maxpool run
    /// @return Matrix (num_pts x 2) of extracted keypoints
    Eigen::MatrixXi maxpool_detect_keypoints(const float* scores_map, int h, int w, int ph, int pv);

    /// @brief Using keypoints' UVs sample descriptor vectors from the descriptor map
    /// @return Matrix (num_pts x channels) of sampled descriptors
    Eigen::MatrixXf sample_descriptors(const float* descriptor_map, const Eigen::MatrixXi& kpts, int c, int h, int w);

private:
    // Super specific maxpool implementation optimized for dkd
    Eigen::MatrixXf maxpool2d(const Eigen::MatrixXf& input, 
                              int h_start, int w_start, 
                              int h_stop, int w_stop, 
                              int radius = 1, int stride = 1);
};

#endif // ALIKE_DKD_H