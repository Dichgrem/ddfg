cmake_minimum_required(VERSION 3.10)
project(myapp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 寻找依赖库
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(nadjieb_mjpeg_streamer REQUIRED)
find_package(yaml-cpp REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/config
    ${PROJECT_SOURCE_DIR}/face
    ${PROJECT_SOURCE_DIR}/timer
)

# 源码文件组织
set(SRC_CONFIG config/ConfigParser.cpp)
set(SRC_FACE face/FaceRecognizer.cpp)
set(SRC_TIMER timer/PerformanceTimer.cpp)

# test_config
add_executable(test_config tests/test_config.cpp ${SRC_CONFIG})
target_link_libraries(test_config yaml-cpp)
target_include_directories(test_config PRIVATE ${PROJECT_SOURCE_DIR}/config)

# test_timer
add_executable(test_timer tests/test_performance_timer.cpp ${SRC_TIMER})
target_include_directories(test_timer PRIVATE ${PROJECT_SOURCE_DIR}/timer)

# test_face
add_executable(test_face tests/test_face_recognition.cpp
    ${SRC_FACE}
    ${SRC_CONFIG}
)
target_link_libraries(test_face
    dlib::dlib
    yaml-cpp
    ${OpenCV_LIBS}
)
target_include_directories(test_face PRIVATE
    ${PROJECT_SOURCE_DIR}/face
    ${PROJECT_SOURCE_DIR}/config
)

# web_capture
add_executable(web_capture web_capture.cpp
    ${SRC_CONFIG}
    ${SRC_FACE}
    ${SRC_TIMER}
)
target_link_libraries(web_capture
    ${OpenCV_LIBS}
    dlib::dlib
    yaml-cpp
    nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
)
target_include_directories(web_capture PRIVATE
    ${PROJECT_SOURCE_DIR}/face
    ${PROJECT_SOURCE_DIR}/config
    ${PROJECT_SOURCE_DIR}/timer
)

