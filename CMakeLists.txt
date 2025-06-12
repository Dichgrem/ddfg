cmake_minimum_required(VERSION 3.10)
project(myapp)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(nadjieb_mjpeg_streamer REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(config)
add_subdirectory(face)
add_subdirectory(timer)

include_directories(
    ${PROJECT_SOURCE_DIR}/config
    ${PROJECT_SOURCE_DIR}/face
    ${PROJECT_SOURCE_DIR}/timer
)

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config yaml-cpp) 

add_executable(test_timer tests/test_performance_timer.cpp)
add_executable(test_face tests/test_face_recognition.cpp)
target_link_libraries(test_face dlib::dlib OpenCV::Core)

add_executable(web_capture web_capture.cpp)
target_link_libraries(web_capture
    dlib::dlib
    OpenCV::Core
    OpenCV::ImgProc
    OpenCV::HighGui
    nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
    yaml-cpp
)

