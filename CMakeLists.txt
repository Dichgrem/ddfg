cmake_minimum_required(VERSION 3.16)
project(FaceRecognitionApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. 寻找依赖库 ---
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
# 建议使用vcpkg或手动编译来安装dlib
# 如果dlib安装在标准位置, find_package可以找到它
# 如果没有，你需要设置 dlib_DIR
find_package(dlib REQUIRED)

# nlohmann_json (header-only, a single file)
# 下载 include/nlohmann/json.hpp 即可
include_directories(${PROJECT_SOURCE_DIR}/include)

# mjpeg-streamer
# 同样，建议使用vcpkg安装或手动编译
# set(nadjieb_mjpeg_streamer_DIR "path/to/cmake/config")
find_package(nadjieb_mjpeg_streamer REQUIRED)


# --- 2. 添加头文件目录 ---
include_directories(${PROJECT_SOURCE_DIR}/include)

# --- 3. 定义我们的核心库 ---
# 将所有模块化的类编译成一个静态库，方便链接
add_library(facerec_core STATIC
    src/ConfigParser.cpp
    src/FaceRecognition.cpp
    src/PerformanceTimer.cpp
)
# 为我们的库添加dlib的依赖
target_link_libraries(facerec_core PUBLIC dlib::dlib)


# --- 4. 定义测试可执行文件 ---
# 测试ConfigParser
add_executable(test_config test/test_config.cpp)
target_link_libraries(test_config PRIVATE facerec_core)

# 测试FaceRecognition
add_executable(test_face_rec test/test_face_rec.cpp)
target_link_libraries(test_face_rec PRIVATE facerec_core ${OpenCV_LIBS})

# --- 5. 定义主程序可执行文件 ---
add_executable(web_capture web_capture.cpp)
target_link_libraries(web_capture PRIVATE 
    facerec_core
    ${OpenCV_LIBS}
    dlib::dlib
    nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
)
