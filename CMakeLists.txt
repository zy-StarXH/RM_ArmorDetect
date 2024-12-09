cmake_minimum_required(VERSION 3.30)

project(RM_Vision)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)

# 设置 OpenCV 的路径
set(OpenCV_DIR D:/OpenCV/opencv/build)  # OpenCV 安装路径
include_directories(${OpenCV_DIR}/include)  # OpenCV 头文件路径
include_directories(D:\\rm_-armor-detect)

# 手动指定库文件路径
link_directories(${OpenCV_DIR}/x64/vc16/lib)

# 设置项目包含的文件
add_executable(RM_Vision
        main.cpp
        armorParam.h
        ArmorParam.cpp
        LightDescriptor.h
        armorDescriptor.h
        armorDescriptor.cpp
        Predictor.h
        Predictor.cpp
)

# 链接 OpenCV 库
target_link_libraries(RM_Vision debug opencv_world4100d.lib optimized opencv_world4100.lib)
