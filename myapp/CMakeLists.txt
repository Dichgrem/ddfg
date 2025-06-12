cmake_minimum_required(VERSION 3.8.0)
project(myapp)

add_subdirectory(../dlib dlib_build)

find_package(OpenCV REQUIRED)
find_package(nadjieb_mjpeg_streamer REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(webcam_face_pose_ex webcam_face_pose_ex.cpp)
#target_link_libraries(webcam_face_pose_ex dlib::dlib ${OpenCV_LIBS})

#add_executable(dnn_face_recognition_ex dnn_face_recognition_ex.cpp)
#target_link_libraries(dnn_face_recognition_ex dlib::dlib ${OpenCV_LIBS})

#basic_capture
add_executable(basic_capture basic_capture.cpp)
target_link_libraries(basic_capture dlib::dlib ${OpenCV_LIBS})

#basic_face_detection
add_executable(basic_face_detect basic_face_detect.cpp)
target_link_libraries(basic_face_detect dlib::dlib ${OpenCV_LIBS})

#capture_detect
add_executable(capture_detect capture_detect.cpp)
target_link_libraries(capture_detect dlib::dlib ${OpenCV_LIBS})

#capture_detect1
add_executable(web_capture1 web_capture1.cpp)
target_link_libraries(web_capture1 dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)

#web_capture
add_executable(web_capture web_capture.cpp)
target_link_libraries(web_capture dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)

#image_to_vector
add_executable(image_to_vector image_to_vector.cpp)
target_link_libraries(image_to_vector dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)

#face_cmp
add_executable(face_cmp face_cmp.cpp)
target_link_libraries(face_cmp dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)

#build_face_lib
add_executable(build_face_lib build_face_lib.cpp)
target_link_libraries(build_face_lib dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)

#face_recognition
add_executable(face_recognition face_recognition.cpp)
target_link_libraries(face_recognition dlib::dlib ${OpenCV_LIBS} nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer)
