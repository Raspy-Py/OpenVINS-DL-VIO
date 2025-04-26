#include "alike_common.h"

std::vector<int64_t> get_tensor_shape(const Ort::Session& session, const std::string& name, bool is_input) {
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_nodes = is_input ? session.GetInputCount() : session.GetOutputCount();
    
    for (size_t i = 0; i < num_nodes; i++) {
        auto current_name = is_input ? 
            session.GetInputNameAllocated(i, allocator) :
            session.GetOutputNameAllocated(i, allocator);
        
        if (name == current_name.get()) {
            auto type_info = is_input ? 
                session.GetInputTypeInfo(i) :
                session.GetOutputTypeInfo(i);
            
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            return tensor_info.GetShape();
        }
    }
    
    throw std::runtime_error("Tensor with name '" + name + "' not found");
}