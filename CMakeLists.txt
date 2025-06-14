cmake_minimum_required(VERSION 3.16)
project(FaceRecognitionApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CTest
enable_testing()

# 开关：只编译并运行 test_config
option(BUILD_ONLY_TEST_CONFIG "只编译并运行 test_config 测试" OFF)

# 公共头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 为 mjpeg-streamer 添加头文件目录 (它是一个纯头文件库)
# 假设您已将 mjpeg-streamer 克隆到 extern/mjpeg-streamer 目录下
include_directories(${PROJECT_SOURCE_DIR}/extern/mjpeg-streamer/include)

# 单独把 ConfigParser 编译成静态库
add_library(config_parser STATIC
    src/ConfigParser.cpp
)
target_include_directories(config_parser PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 如果只需要编译 test_config
if (BUILD_ONLY_TEST_CONFIG)
    message(STATUS ">>> BUILD_ONLY_TEST_CONFIG 模式: 仅编译 test_config <<<")

    add_executable(test_config
        test/test_config.cpp
    )
    target_link_libraries(test_config
        PRIVATE config_parser
    )
    add_test(NAME test_config COMMAND test_config)
    return()
endif()

# 常规模式：查找外部依赖
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

# 核心库：facerec_core
# 移除 src/ConfigParser.cpp，因为它已经在 config_parser 库中编译了
add_library(facerec_core STATIC
    src/FaceRecognition.cpp
    src/PerformanceMonitor.cpp
)
target_include_directories(facerec_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(facerec_core
    PUBLIC
        config_parser # facerec_core 依赖于 config_parser，通过 PUBLIC 传递依赖
        dlib::dlib
        ${OpenCV_LIBS}
)

# 测试1：test_config.cpp
add_executable(test_config
    test/test_config.cpp
)
target_link_libraries(test_config
    PRIVATE config_parser
)
add_test(NAME test_config COMMAND test_config)

# 测试2：test_face_rec.cpp
add_executable(test_face_rec
    test/test_face_rec.cpp
)
target_link_libraries(test_face_rec
    PRIVATE
        facerec_core
        dlib::dlib # 虽然 facerec_core 已经链接了 dlib，但这里再次明确链接通常是安全的，以防万一
        ${OpenCV_LIBS} # 同上，再次明确链接 OpenCV 库
)
add_test(NAME test_face_rec COMMAND test_face_rec)

# 主程序 web_capture
add_executable(web_capture web_capture.cpp)
target_link_libraries(web_capture
    PRIVATE
        facerec_core
        dlib::dlib # 同上，明确链接
        ${OpenCV_LIBS} # 同上，明确链接
        # mjpeg-streamer 是纯头文件库，不需要在这里链接
)

# --- 模型和配置文件的复制 (可选但推荐) ---
# 这确保您的可执行文件在运行时能找到它们。
# 目标路径是相对于构建目录的。

# 复制 dlib 模型
file(GLOB_RECURSE DLIB_MODELS "${PROJECT_SOURCE_DIR}/model/*.dat")
file(COPY ${DLIB_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/model)

# 复制人脸库 (facelib 内容)
file(COPY "${PROJECT_SOURCE_DIR}/facelib/" DESTINATION "${CMAKE_BINARY_DIR}/facelib/")

# 复制 config.json
file(COPY "${PROJECT_SOURCE_DIR}/config/config.json" DESTINATION "${CMAKE_BINARY_DIR}/config")
