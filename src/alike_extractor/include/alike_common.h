#ifndef ALIKE_ALIKE_COMMON_H
#define ALIKE_ALIKE_COMMON_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct ALikeMetadata{
    // factor, by which the last input image was scaled
    float scale_factor; 

    // horizontal padding height added to the input image (top and bottom)
    int h_padding; 
    int v_padding; // vertical padding (left and right)

    // dimensions of ALike CNN input
    int input_rows; 
    int input_cols; 

    // descriptor dimensionality
    int channels;
};

std::vector<int64_t> get_tensor_shape(const Ort::Session& session, const std::string& name, bool is_input);


#endif // ALIKE_ALIKE_COMMON_H