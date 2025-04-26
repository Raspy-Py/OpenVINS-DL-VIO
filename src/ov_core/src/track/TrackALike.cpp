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

#include "TrackALike.h"

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

//pls work
//#include <opencv2/xfeatures2d.hpp>

#include <vector>
#include <utility>
#include <algorithm>

using namespace ov_core;

// Helper function to sort data based on ids
void sortData(std::vector<size_t>& ids, std::vector<cv::KeyPoint>& keypoints, Eigen::MatrixXf& descriptors) {
  // Sort indices, then rearange data based on them
  std::vector<size_t> indices(ids.size());
  for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&ids](size_t a, size_t b) {
      return ids[a] < ids[b];
  });

  std::vector<size_t> sortedIds(ids.size());
  std::vector<cv::KeyPoint> sortedKeypoints(keypoints.size());
  Eigen::MatrixXf sortedDescriptors(descriptors.rows(), descriptors.cols());

  for (size_t i = 0; i < indices.size(); ++i) {
      sortedIds[i] = ids[indices[i]];
      sortedKeypoints[i] = keypoints[indices[i]];
      sortedDescriptors.row(i) = descriptors.row(indices[i]);
  }

  ids.swap(sortedIds);
  keypoints.swap(sortedKeypoints);
  descriptors.swap(sortedDescriptors);
}

void TrackALike::GMS_match_refine(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches1to2, std::vector<cv::DMatch>& matchesGMS) {
  auto img_size = img_last[0].size();
  // cv::xfeatures2d::matchGMS(img_size, img_size, pts1, pts2, matches1to2, matchesGMS, false, false, 6.0);
  //matchesGMS = matches1to2;
}

TrackALike::TrackALike(
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo, HistogramMethod histmethod, 
  const std::string& model_path, bool use_mask, int num_pts, int radius, int padding, float match_threshold)  
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod),
        num_pts(num_pts), match_threshold(match_threshold) {
  alike = std::make_unique<ALike>(model_path, use_mask);
  dkd = std::make_unique<DKD>(num_pts, radius, padding);
}


void TrackALike::feed_new_camera(const CameraData &message) {
  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    // Save!
    img_curr[cam_id] = img;
  }

  // Only monocular tracking is supported
  feed_monocular(message, 0);
}

void TrackALike::feed_monocular(const CameraData &message, size_t msg_id) {
  // Lock new feeds, while current image is being processed
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image and corresponding mask
  cv::Mat img = img_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2BGR);
  rT2 = boost::posix_time::microsec_clock::local_time(); // image fed and read

  // If didn't have any successful tracks last time, just extract this time
  if (pts_last[cam_id].empty()){
    std::vector<cv::KeyPoint> pts;
    std::vector<size_t> ids;
    Eigen::MatrixXf descriptors;
    perform_detection_monocular(rgb_img, mask, pts, descriptors);

    // Save current image
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = pts;
    descriptors_last[cam_id] = descriptors;

    // Assign unique ids to freshly detected features
    ids_last[cam_id].resize(num_pts);
    for (int i = 0; i < num_pts; i++) {
      ids_last[cam_id][i] = ++currid;
    }
    return;
  }

  // In Alike detections are performed only on the current image:
  std::vector<cv::KeyPoint> pts_new;
  Eigen::MatrixXf descriptors_new;
  perform_detection_monocular(rgb_img, mask, pts_new, descriptors_new);
  rT3 = boost::posix_time::microsec_clock::local_time(); // ALike + DKD

  // Matching the points using cosine distance
  std::vector<cv::DMatch> matches;
  perform_matching(descriptors_last[cam_id], descriptors_new, matches);
  std::vector<cv::DMatch> refined_matches;

  //GMS_match_refine(pts_last[cam_id], pts_new, matches, refined_matches);
  refined_matches = std::move(matches);

  assert(pts_new.size() == ids_last[cam_id].size());
  PRINT_ERROR(RED "[ALIKE-INFO]:  matches: %.zu\n" RESET, refined_matches.size());

  rT4 = boost::posix_time::microsec_clock::local_time(); // Cosine distance matching
  
  // FIXME: this probably never happens with ALike
  const size_t min_matches = 8;
  if (refined_matches.size() < min_matches && false) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[ALike-EXTRACTOR]: Failed to get enough matches, resetting.....\n" RESET);
    return;
  }

  // Here we assign old ids from old points to corresponding new points
  // and generate new ids for those, which were not matched
  std::vector<size_t> ids_new(num_pts, 0);
  ids_new.reserve(num_pts);
  for (auto match : refined_matches) {
    ids_new[match.trainIdx] = ids_last[cam_id][match.queryIdx];
  }
  
  for(size_t& id : ids_new){
    if (id == 0) id = ++currid;
  }
  
  // Sort by ids
  sortData(ids_new, pts_new, descriptors_new);

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = pts_new;
    ids_last[cam_id] = ids_new;
    descriptors_last[cam_id] = descriptors_new;
  }

  rT5 =boost::posix_time::microsec_clock::local_time(); // All Done

  // Update our feature database, with these new observations
  // Only matched keypoints are updated
  for (size_t i = 0; i < refined_matches.size(); i++){
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(pts_new.at(i).pt);
    database->update_feature(
      ids_new.at(i), message.timestamp, cam_id, 
      pts_new.at(i).pt.x, pts_new.at(i).pt.y, // Real UVs
      npt_l.x, npt_l.y); // Undistorted UVs
  }

  // PRINT_DEBUG(RED "[TIME-ALIKE]: %.4fms Image fed and read\n" RESET, (rT2 - rT1).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-ALIKE]: %.4fms ALike + DKD \n" RESET, (rT3 - rT2).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-ALIKE]: %.4fms Cosine similarity matching \n" RESET, (rT4 - rT3).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-ALIKE]: %.4fms Sorting and postprocess \n" RESET, (rT5 - rT4).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-ALIKE]: %.4fms for total\n" RESET, (rT5 - rT1).total_microseconds() * 1e-3);
}

void TrackALike::perform_detection_monocular(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &pts0, Eigen::MatrixXf& descriptors) {
  Eigen::MatrixXi keypoints;

  // Run CNN to extarct scores and descriptor maps
  alike->run(img, mask);

  // Sample keypoints from extracted maps
  dkd->run(
    alike->get_scores_map(), 
    alike->get_descriptor_map(), 
    alike->get_meta(),
    keypoints,
    descriptors
  );

  // OpenVINS' feature database uses cv::KeyPoint
  pts0.reserve(keypoints.rows());
  for (int i = 0; i < keypoints.rows(); ++i) {
    cv::KeyPoint pt;
    pt.pt.x = keypoints(i, 0);
    pt.pt.y = keypoints(i, 1);
    pts0.push_back(pt);
  }
}

// Mutual nearest neighbors
// void TrackALike::perform_matching(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<std::pair<int, int>> &matches) {
void TrackALike::perform_matching(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<cv::DMatch>& matches) {
  Eigen::MatrixXf sim = desc1 * desc2.transpose();
  sim = (sim.array() >= match_threshold).select(sim, Eigen::MatrixXf::Zero(sim.rows(), sim.cols()));

  Eigen::VectorXi nn12 = Eigen::VectorXi::Zero(sim.rows());
  Eigen::VectorXi nn21 = Eigen::VectorXi::Zero(sim.cols());

  for (int i = 0; i < sim.rows(); ++i) {
    sim.row(i).maxCoeff(&nn12(i));
  }

  for (int i = 0; i < sim.cols(); ++i) {
    sim.col(i).maxCoeff(&nn21(i));
  }

  // for (int i = 0; i < sim.rows(); ++i) {
  //   if (i == nn21[nn12[i]] && sim.row(i)[nn12[i]]){
  //     matches.emplace_back(i, nn12[i]);
  //   }
  // }

  for (int i = 0; i < sim.rows(); ++i) {
    if (i == nn21[nn12[i]] && sim.row(i)[nn12[i]]){
      matches.emplace_back(i, nn12[i], sim(i, nn12[i]));
    }
  }
}

