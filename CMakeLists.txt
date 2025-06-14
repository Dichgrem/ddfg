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
find_package(dlib    REQUIRED)

# 核心库：facerec_core
add_library(facerec_core STATIC
    src/ConfigParser.cpp
    src/FaceRecognition.cpp
    src/PerformanceTimer.cpp
)
target_include_directories(facerec_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(facerec_core
    PUBLIC config_parser
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
    PRIVATE facerec_core
            dlib::dlib
            ${OpenCV_LIBS}
)
add_test(NAME test_face_rec COMMAND test_face_rec)

# （可选）主程序
# add_executable(web_capture web_capture.cpp)
# target_link_libraries(web_capture
#     PRIVATE facerec_core
#             dlib::dlib
#             ${OpenCV_LIBS}
# )

