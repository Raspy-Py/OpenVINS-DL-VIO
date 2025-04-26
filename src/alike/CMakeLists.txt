cmake_minimum_required(VERSION 3.16)

project(alike_extractor)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
#set(CMAKE_BUILD_TYPE Release)

set(LIB_NAME alike_extractorlib)
set(PROJ_NAME alike_extractor)

# Whether to build a test app
set(BUILD_ALIKE_TEST true)


# Enable optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIBRARIES onnxruntime)
link_directories(
    ${ONNXRUNTIME_ROOT_PATH}/lib
)

# CUDA
find_package(CUDA REQUIRED)

# OpenCV
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
message(STATUS "OPENCV: " ${OpenCV_VERSION})

# Eigen
find_package(Eigen3 REQUIRED)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(INCLUDE_DIRS
    ${PROJECT_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
include_directories(${INCLUDE_DIRS})


# Catkin build
find_package(catkin REQUIRED 
    COMPONENTS roscpp
)
catkin_package(
        CATKIN_DEPENDS roscpp
        INCLUDE_DIRS include ${INCLUDE_DIRS}
        LIBRARIES ${LIB_NAME}
)

# Project sources
file(GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB LIB_HEADERS ${CMAKE_CURRENT_LIST_DIR}/include/*.h)


if (BUILD_ALIKE_TEST)
    message(WARNING "Bulding test")

    set(ALIKE_TEST_SRC ${LIB_SOURCES} ${LIB_HEADERS} alike_test.cpp)
    add_executable(alike_test ${ALIKE_TEST_SRC})

    target_link_libraries(alike_test
        ${ONNXRUNTIME_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES})
endif ()

add_library(${LIB_NAME} SHARED  ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${LIB_NAME}
    ${ONNXRUNTIME_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES})

# Add ONNX Runtime library path to runtime library search path
set_target_properties(${LIB_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib"
)

install(TARGETS ${LIB_NAME} ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES ${LIB_HEADERS} DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})