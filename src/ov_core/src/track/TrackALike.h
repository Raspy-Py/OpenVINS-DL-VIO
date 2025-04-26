/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_ALIKE_H
#define OV_CORE_TRACK_ALIKE_H

#include "TrackBase.h"

#include "alike_extractor/dkd.h"
#include "alike_extractor/alike.h"
#include "alike_extractor/alike_common.h"

#include <Eigen/Dense>
#include <memory>
#include <fstream>


namespace ov_core {

/**
 * @brief Alike tracking of features.
 *
 * This implementation uses ALike for both keypoint detection and feature extraction, 
 * which results in keypoint-descriptor pairs. Those pairs can be matched across
 * different frames using cosine similarity.
 */
class TrackALike : public TrackBase {
public:
  struct Options {
    // Base options
    std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras;
    int numfeats;
    int numaruco;
    bool stereo;
    HistogramMethod histmethod;
    // ALike options
    std::string model_path;
    bool use_mask;
    int num_pts;
    int radius;
    int padding;
    float match_threshold;
  };

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   */
  TrackALike(
    // Base options
    std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo, HistogramMethod histmethod, 
    // ALike options
     const std::string& model_path, bool use_mask, int num_pts, int radius, int padding, float match_threshold);

  TrackALike(const Options& options) 
    : TrackBase(options.cameras, options.numfeats, options.numaruco, options.stereo, options.histmethod),
      num_pts(options.num_pts), radius(options.radius), padding(options.padding), match_threshold(options.match_threshold),
      alike(std::make_unique<ALike>(options.model_path, options.use_mask)),
      dkd(std::make_unique<DKD>(num_pts, radius, padding)) {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Detects new features in the current image
   * @param img0 image we will detect features on 
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param descriptors matrix of feature descriptors for each currently extracted keypoint
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &pts0, Eigen::MatrixXf& descriptors);

  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  // void perform_matching(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<std::pair<int, int>> &matches);
  void perform_matching(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<cv::DMatch>& matches);

  /**
    * @brief Refine matches using GMS
   */
  void GMS_match_refine(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches1to2, std::vector<cv::DMatch>& matchesGMS);

private:
  // Per frame data
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, Eigen::MatrixXf> descriptors_last;

  int num_pts;
  int radius;
  int padding;
  float match_threshold;

  // ALike stuff
  std::unique_ptr<ALike> alike;
  std::unique_ptr<DKD> dkd;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_ALIKE_H */
