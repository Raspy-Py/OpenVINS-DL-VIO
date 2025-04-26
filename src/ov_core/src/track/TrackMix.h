#ifndef OV_CORE_TRACK_MIX_H
#define OV_CORE_TRACK_MIX_H

#include "TrackBase.h"

#include "alike_extractor/dkd.h"
#include "alike_extractor/alike.h"
#include "alike_extractor/alike_common.h"

#include <Eigen/Dense>
#include <memory>
#include <fstream>


namespace ov_core {

/**
 * @brief Mix tracking of features.
 *
 * This implementation uses Alike for both keypoint detection and feature extraction, 
 * which results in keypoint-descriptor pairs. Those pairs can be matched across
 * different frames using cosine similarity.
 */
class TrackMix : public TrackBase {
public:
  struct Options {
    // Base options
    std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras;
    int numfeats;
    int numaruco;
    bool stereo;
    HistogramMethod histmethod;
    // KLT options
    int fast_threshold;
    int gridx;
    int gridy;
    int minpxdist;
    // ALike options
    std::string model_path;
    bool use_mask;
    int num_pts;
    int radius;
    int padding;
    float match_threshold;
  };


public:
  TrackMix(const Options& options)
  : TrackBase(options.cameras, options.numfeats, options.numaruco, options.stereo, options.histmethod),
    threshold(options.fast_threshold), grid_x(options.gridx), grid_y(options.gridy), min_px_dist(options.minpxdist),
    num_pts(options.num_pts), radius(options.radius), padding(options.padding), match_threshold(options.match_threshold),
    alike(std::make_unique<ALike>(options.model_path, options.use_mask)),
    dkd(std::make_unique<DKD>(num_pts, radius, padding))
  {}

  TrackMix(
    // Base options
    std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo, HistogramMethod histmethod, 
    // KLT options
    int fast_threshold, int gridx, int gridy, int minpxdist,
    // ALike options
    const std::string& model_path, bool use_mask, int num_pts, int radius, int padding, float match_threshold);

  void feed_new_camera(const CameraData &message) override;

protected:
  void feed_monocular(const CameraData &message, size_t msg_id);

  void detect_ALike(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::Point>& keypoints, Eigen::MatrixXf& descriptors);
  void match_ALike(const Eigen::MatrixXf& desc1, const Eigen::MatrixXf& desc2, std::vector<cv::DMatch>& matches);

  void detect_FAST(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0);
  void match_KLT(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);



  void GMS_match_refine(std::vector<cv::KeyPoint> pts1, std::vector<cv::KeyPoint> pts2, std::vector<cv::DMatch> matches1to2, std::vector<cv::DMatch>& matchesGMS);
  
  // Runs ALike's pipeline to estimate the initial affine transformation between two images
  cv::Mat  estimate_initial_affine(cv::Mat& img, cv::Mat& mask);

private:
  /*
  * ALike stuff
  */
  int num_pts;
  int radius;
  int padding;
  float match_threshold;

  std::vector<cv::Point> alike_pts_last;
  Eigen::MatrixXf alike_desc_last;
  std::mutex mtx_alike_last_vars;
  std::unique_ptr<ALike> alike;
  std::unique_ptr<DKD> dkd;

  /*
  * KLT stuff
  */
  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_MIX_H */
