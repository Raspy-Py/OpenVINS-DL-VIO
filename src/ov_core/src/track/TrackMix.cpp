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

#include "TrackMix.h"

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

//pls work
// #include <opencv2/xfeatures2d.hpp>

#include <vector>
#include <utility>
#include <algorithm>

using namespace ov_core;


void TrackMix::GMS_match_refine(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches1to2, std::vector<cv::DMatch>& matchesGMS) {
  auto img_size = img_last[0].size();
  //cv::xfeatures2d::matchGMS(img_size, img_size, pts1, pts2, matches1to2, matchesGMS, false, false, 6.0);
  //matchesGMS = matches1to2;
}

TrackMix::TrackMix(
  // Base options
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo, HistogramMethod histmethod, 
  // KLT options
  int fast_threshold, int gridx, int gridy, int minpxdist,
  // ALike options
  const std::string& model_path, bool use_mask, int num_pts, int radius, int padding, float match_threshold)  
    : 
    TrackBase(cameras, numfeats, numaruco, stereo, histmethod),
    threshold(fast_threshold), grid_x(gridx), grid_y(gridy), min_px_dist(minpxdist),
    num_pts(num_pts), match_threshold(match_threshold)
{
  alike = std::make_unique<ALike>(model_path, use_mask);
  dkd = std::make_unique<DKD>(num_pts, radius, padding);
}


void TrackMix::feed_new_camera(const CameraData &message) {
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

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Only monocular tracking is supported
  feed_monocular(message, 0);
}


static void draw_affine_indicator(cv::Mat& img, cv::Mat& affine) {
  auto img_size = img.size();
  float ar = (float)img_size.width / (float)img_size.height;
  // Background for indicator
  int bg_width = 200;
  int bg_height = (float)bg_width / ar;
  cv::rectangle(img, cv::Point(0, 0), cv::Point(bg_width, bg_height), cv::Scalar(255, 255, 255), -1);

  // Draw the affine transformation
  int ind_width = bg_width / 2;
  int ind_height = bg_height / 2;

  // Original rectangle
  std::vector<cv::Point> rect_corners = {
    cv::Point(-ind_width / 2, -ind_height / 2),
    cv::Point(ind_width / 2, -ind_height / 2),
    cv::Point(ind_width / 2, ind_height / 2),
    cv::Point(-ind_width / 2, ind_height / 2)
  };

  // Affine transformed rectangle
  std::vector<cv::Point> rect_corners_affine;
  for (auto& pt : rect_corners) {
    cv::Point2f npt = cv::Point2f(
      affine.at<double>(0, 0) * pt.x + affine.at<double>(0, 1) * pt.y + affine.at<double>(0, 2),
      affine.at<double>(1, 0) * pt.x + affine.at<double>(1, 1) * pt.y + affine.at<double>(1, 2)
    );
    rect_corners_affine.push_back(cv::Point(npt.x, npt.y));
  }

  // Translate to the center of indicator area
  for (size_t i = 0; i < rect_corners.size(); i++) {
    rect_corners[i].x += bg_width / 2;
    rect_corners[i].y += bg_height / 2;
    rect_corners_affine[i].x += bg_width / 2;
    rect_corners_affine[i].y += bg_height / 2;
  }
  
  cv::polylines(img, rect_corners, true, cv::Scalar(100), 1);
  cv::polylines(img, rect_corners_affine, true, cv::Scalar(0), 1);
}

void TrackMix::feed_monocular(const CameraData &message, size_t msg_id) {
  /*
  1. Estimate initial affine transformation using ALike
  2. Transform old keypoints using the affine transformation
  3. Perform detection using FAST with transformed keypoints
  ...
  n. Other stuff is identicall to the original TarckKLT::feed_monocular
  */



  // Lock new feeds, while current image is being processed
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image and corresponding mask
  cv::Mat img = img_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat img_rgb;
  cv::Mat debug_img = img.clone();
  cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);


  // #1
  cv::Mat estimated_affine = estimate_initial_affine(img_rgb, mask);
  // print estimated_affine
  std::cout << "estimated_affine: \n" << estimated_affine << std::endl;
  draw_affine_indicator(debug_img, estimated_affine);


  rT2 = boost::posix_time::microsec_clock::local_time();


  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    detect_FAST(imgpyr, mask, good_left, good_ids_left);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    return;
  }


  // #2
  int pts_before_detect = (int)pts_last[cam_id].size();
  auto pts_left_old = pts_last[cam_id];
  auto ids_left_old = ids_last[cam_id];
  detect_FAST(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  auto pts_left_new = pts_left_old;
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    cv::Point2f pt = pts_left_new.at(i).pt;
    cv::Point2f npt = camera_calib.at(cam_id)->undistort_cv(pt);
    cv::Point2f npt_affine = cv::Point2f(
      estimated_affine.at<double>(0, 0) * npt.x + estimated_affine.at<double>(0, 1) * npt.y + estimated_affine.at<double>(0, 2),
      estimated_affine.at<double>(1, 0) * npt.x + estimated_affine.at<double>(1, 1) * npt.y + estimated_affine.at<double>(1, 2)
    );
    pts_left_new.at(i).pt = camera_calib.at(cam_id)->distort_cv(npt_affine);
  }

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  //std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  // Lets track temporally
  // printf("old|new pts size: %d|%d\n", pts_left_old.size(), pts_left_new.size());
  match_KLT(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
  assert(pts_left_new.size() == ids_left_old.size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[Mix-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = debug_img;
    // img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // PRINT_DEBUG(RED "[TIME-Mix]: %.4fms initial affine estimate\n" RESET, (rT2 - rT1).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-Mix]: %.4fms FAST detection \n" RESET, (rT3 - rT2).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-Mix]: %.4fms KLT matching \n" RESET, (rT4 - rT3).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-Mix]: %.4fms sorting and postprocess \n" RESET, (rT5 - rT4).total_microseconds() * 1e-3);
  // PRINT_DEBUG(RED "[TIME-Mix]: %.4fms for total\n" RESET, (rT5 - rT1).total_microseconds() * 1e-3);
}


cv::Mat TrackMix::estimate_initial_affine(cv::Mat& img, cv::Mat& mask) {
  /*
  1. Detect keypoints with semantics
  2. Match keypoints
  3. Estimate affine matrix
  */

  // #1
  std::vector<cv::Point> alike_pts_new;
  std::vector<size_t> ids;
  Eigen::MatrixXf alike_desc_new;
  detect_ALike(img, mask, alike_pts_new, alike_desc_new);

  // Undistort keypoints for correct affine approximation
  for (auto& new_pt : alike_pts_new) {
    new_pt = camera_calib.at(0)->undistort_cv(new_pt);
  }

  if ( alike_pts_last.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_alike_last_vars);
    alike_pts_last = alike_pts_new;
    alike_desc_last = alike_desc_new;
    return cv::Mat::eye(2, 3, CV_64F);
  }

  // #2
  std::vector<cv::DMatch> alike_matches;
  match_ALike(alike_desc_last, alike_desc_new, alike_matches);

  // Not enough matches to estimate affine matrix
  if (alike_matches.size() < 8) {
    PRINT_DEBUG(YELLOW "Not enough matches (%zu) to make an initial affine estimate\n" RESET, alike_matches.size());
    return cv::Mat::eye(2, 3, CV_64F);
  }

  // TODO: [optional] GMS step
  // std::vector<cv::DMatch> refined_matches;

  // #3
  std::vector<cv::Point2f> kpts_prev, kpts_curr;
  kpts_prev.reserve(alike_matches.size());
  kpts_curr.reserve(alike_matches.size());
  for (auto match : alike_matches) {
    kpts_prev.push_back(alike_pts_last[match.queryIdx]);
    kpts_curr.push_back(alike_pts_new[match.trainIdx]);
  }
  cv::Mat affine_matrix = cv::estimateAffine2D(kpts_prev, kpts_curr);
  if (affine_matrix.empty()) {
    PRINT_DEBUG(YELLOW "Affine estimation failed. Returning identity transformation.\n" RESET, alike_matches.size());
    affine_matrix = cv::Mat::eye(2, 3, CV_64F);
}
  { // Update last vars
    std::lock_guard<std::mutex> lckv(mtx_alike_last_vars);
    alike_pts_last = alike_pts_new;
    alike_desc_last = alike_desc_new;
  }

  return affine_matrix;    
}


void TrackMix::detect_ALike(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::Point> &keypoints, Eigen::MatrixXf& descriptors) {
  // Run CNN to extract scores and descriptor maps
  alike->run(img, mask);

  Eigen::MatrixXi keypoints_buffer;
  // Sample keypoints from extracted maps
  dkd->run(
    alike->get_scores_map(), 
    alike->get_descriptor_map(), 
    alike->get_meta(),
    keypoints_buffer,
    descriptors
  );

  // Unpack keypoints from a buffer
  keypoints.reserve(keypoints_buffer.rows());
  for (int i = 0; i < keypoints_buffer.rows(); ++i) {
    cv::Point pt;
    pt.x = keypoints_buffer(i, 0);
    pt.y = keypoints_buffer(i, 1);
    keypoints.push_back(pt);
  }
}

// Mutual nearest neighbors
// void TrackMix::perform_matching(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<std::pair<int, int>> &matches) {
void TrackMix::match_ALike(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<cv::DMatch>& matches) {
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

  for (int i = 0; i < sim.rows(); ++i) {
    if (i == nn21[nn12[i]] && sim.row(i)[nn12[i]]){
      matches.emplace_back(i, nn12[i], sim(i, nn12[i]));
    }
  }
}


void TrackMix::detect_FAST(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}



void TrackMix::match_KLT(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                        std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) 
{

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

  //cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}
