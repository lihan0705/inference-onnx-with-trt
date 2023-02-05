cmake_minimum_required(VERSION 3.23)

project(mnsitClassification)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set( CMAKE_CXX_FLAGS "-std=c++11" )
set( CMAKE_BUILD_TYPE "Debug" )
# include_directories( "/usr/include/eigen3" )
# add_library(irfusion comfunc.c post_sins_gnss.cpp)
set(TENSORRT_INCLUDE_DIRS  /home/liangdao_hanli/software/TensorRT-8.2.5.1/include)
set(TENSORRT_LIBRARY_DIRS  /home/liangdao_hanli/software/TensorRT-8.2.5.1/lib)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)


link_directories(
	${TENSORRT_LIBRARY_DIRS}
	/usr/lib/aarch64-linux-gnu
  /usr/aarch64-linux-gnu/lib/
)

set (source_dir "${PROJECT_SOURCE_DIR}/src/")

file (GLOB source_files "${source_dir}/*.cpp")

add_executable (mnsitClassification main.cpp ${source_files})

target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    ${OpenCV_LIBRARIES}
)
