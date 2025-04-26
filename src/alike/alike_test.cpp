#include <vector>
#include <iostream>
#include <chrono>

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include <string>
#include <filesystem>

#include "dkd.h"
#include "alike.h"
#include "alike_common.h"


int main(int argc, char** argv) {
    if (argc < 4){
        std::cout << "Not enough arguments: " << std::endl
                  << "./alike_test <model_path> <input_image> <output_folder>" << std::endl;
        return -1;
    }   

    std::string model_path{argv[1]};
    std::string input_image_path{argv[2]};
    std::string output_folder{argv[3]};


    // Prepare data
    cv::Mat input_image = cv::imread(input_image_path);
    auto image_shape = input_image.size;

    // Load model
    ALike alike(model_path);
    DKD dkd(400, 2, 2);

    // Run CNN
    alike.run(input_image, cv::Mat{});

    float* scores_map = alike.get_scores_map();
    float* descriptor_map = alike.get_descriptor_map();
    auto meta = alike.get_meta();

    // Save score map
    int height = meta.input_rows, width = meta.input_cols;
    cv::Mat output_image(height, width, CV_32FC1, scores_map);  
    output_image *= 255.0f;

    cv::cvtColor(output_image, output_image, cv::COLOR_GRAY2BGR);
    cv::imwrite(output_folder + "/scores_map.png", output_image);
    cv::resize(output_image, output_image, cv::Size(image_shape[1], image_shape[0]), 0, 0, cv::INTER_LINEAR);


    // Testing the rest of the pipeline
    Eigen::MatrixXi keypoints;

    Eigen::MatrixXf descriptors;  
    dkd.run(scores_map, descriptor_map, meta, keypoints, descriptors);

    cv::Mat results_image = input_image.clone();

    std::cout << "Detected " << keypoints.rows() << " keypoints." << std::endl;
    for (int i = 0; i < keypoints.rows(); ++i) {
        int x = keypoints(i, 0);   
        int y = keypoints(i, 1);
        cv::circle(results_image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
        cv::circle(output_image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite(output_folder + "/results_image.png", results_image);
    cv::imwrite(output_folder + "/labeled_score_map.png", output_image);

    return 0;
}