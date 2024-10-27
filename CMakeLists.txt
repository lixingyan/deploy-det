cmake_minimum_required(VERSION 3.11)
project(CountObj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/usr/local/cuda/include)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(build
    main11.cpp
    TrtModel.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    # ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)

target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")
target_link_libraries(build PRIVATE ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})





# cmake_minimum_required(VERSION 3.11)
# project(build LANGUAGES CXX)

# # Set C++ standard to 14
# set(CMAKE_CXX_STANDARD 14)
# set(TensorRT_DIR "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1")  # 替换成你的TensorRT安装路径
# set(TensorRT_LIBRARIES "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/lib/libnvinfer.so")  # 替换成你实际的TensorRT库文件路径


# # Set the path to the TensorRT installation
# set(TensorRT_DIR "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/lib/cmake/TensorRT" CACHE PATH "Path to TensorRT Config file")

# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11/")   # CUDA安装根目录
# set(TensorRT_ROOT "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/include")       # TensorRT安装根目录

# find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
# # Include directories
# include_directories("/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/include")
# # include_directories(/usr/local/cuda/include)
# include_directories("/usr/local/cuda-11/include")
# include_directories("/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/samples/common")
# link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# # Link directories
# link_directories("/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/lib" )
# # Add executable
# add_executable(build
#     main11.cpp  
#     TrtModel.cpp  
#     /home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/samples/common/logger.cpp  
#     /home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/samples/common/sampleUtils.cpp 
# )

# # Set CUDA architectures
# target_include_directories(build PRIVATE 
#                             "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/include" 
#                            "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/samples/common")

# set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# target_link_libraries(build PRIVATE ${OpenCV_LIBS} 
#                                     ${TensorRT_LIBRARIES} 
#                                     ${CUDA_LIBRARIES})`   







# cmake_minimum_required(VERSION 3.11)
# project(build LANGUAGES CXX CUDA)

# # Set C++ standard to 14
# set(CMAKE_CXX_STANDARD 14)

# # Set the path to the TensorRT installation
# set(TensorRT_ROOT "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/")
# # set(TensorRT_DIR "/home/xiashu/tensorRT8.5.3/TensorRT-8.5.3.1/")

# # Set the path to the CUDA toolkit installation
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11/")   # CUDA安装根目录

# # Find required packages
# find_package(OpenCV REQUIRED)
# # find_package(TensorRT REQUIRED)

# # Include directories
# include_directories(
#     ${OpenCV_INCLUDE_DIRS}
#     ${TensorRT_INCLUDE_DIRS}
#     ${CUDA_TOOLKIT_ROOT_DIR}/include
#     ${TensorRT_ROOT}/samples/common   # 添加TensorRT的common目录
# )

# # Link directories
# link_directories(
#     ${CUDA_TOOLKIT_ROOT_DIR}/lib64
#     ${TensorRT_ROOT}/lib
# )

# # Add executable
# add_executable(build
#     main11.cpp  
#     TrtModel.cpp  
#     ${TensorRT_ROOT}/samples/common/logger.cpp  
#     ${TensorRT_ROOT}/samples/common/sampleUtils.cpp 
# )

# # Set CUDA architectures
# set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# # Link libraries
# target_link_libraries(build PRIVATE 
#     ${OpenCV_LIBS}
#     ${TensorRT_LIBRARIES}
#     ${CUDA_LIBRARIES}
# )
