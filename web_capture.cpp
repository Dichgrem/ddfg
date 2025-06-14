#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h> // for extract_image_chip, get_face_chip_details
// #include <dlib/image_processing/render_face_detections.h> // 移除此行，因为它可能引入GUI依赖
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/console_progress_indicator.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem> // C++17 filesystem，用于遍历人脸库目录

// 项目自定义头文件
#include "ConfigParser.h"    // 位于 include/
#include "FaceRecognition.hpp" // 位于 include/

// MJPEG Streamer 的头文件路径
#include <nadjieb/mjpeg_streamer.hpp> // 确保这个路径和文件存在

namespace fs = std::filesystem; // 使用 std::filesystem 命名空间

int main() {
    // --- 初始化 ConfigParser ---
    ConfigParser config; // 默认构造
    // 尝试加载配置文件
    if (!config.load("config/config.json")) { // 调用 load 方法
        std::cerr << "错误: 无法加载配置文件 config/config.json" << std::endl;
        return 1;
    }
    config.printAll(); // 打印所有配置，方便调试

    // --- 初始化 FaceRecognition 对象 ---
    // 将已加载的 config 对象传递给 FaceRecognition 构造函数
    FaceRecognition face_recognizer(config);

    try {
        // 根据您提供的 FaceRecognition.hpp，模型和人脸库的加载应该发生在
        // FaceRecognition 的构造函数内部（通过调用其私有方法 loadModels 和 buildFaceLibrary）。
        // 因此，这里不再需要显式调用 loadModels 或 loadFaceDatabase。
        //
        // 但是，FaceRecognition.hpp 中没有公共方法来获取人脸库大小、阈值等，
        // 只有 printFaceLibInfo()。
        face_recognizer.printFaceLibInfo(); // 打印人脸库信息，用于确认加载状态

    } catch (const std::exception& e) {
        std::cerr << "人脸识别初始化失败: " << e.what() << std::endl;
        return 1;
    }

    cv::VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 初始化 MJPEG Streamer
    // 使用完整命名空间 nadjieb::mjpeg_streamer::mjpeg_streamer
    nadjieb::MJPEGStreamer streamer;
    streamer.start(8080); // 在 8080 端口启动流

    // Dlib 人脸检测器 (通常 FaceRecognition 内部会有一个，但这里为简化流程，可以独立声明)
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);

        // 检测人脸
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        // 处理每个检测到的人脸
        for (const auto& face_rect : faces) {
            // --- 校正后识别出人脸 ---
            // 在发送给识别前，提取并对齐人脸芯片
            // 使用 FaceRecognition 对象获取形状预测器
            dlib::full_object_detection shape = face_recognizer.getShapePredictor()(dlib_img, face_rect);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

            // 调用 FaceRecognition 的 recognize 方法
            std::string recognized_name = face_recognizer.recognize(face_chip);

            // --- 在图像上绘制人脸信息 ---
            cv::Scalar color = (recognized_name == "Stranger") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0); // 陌生人红色，已知人脸绿色
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(recognized_name, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
            // 显式转换为 int
            cv::Point textOrg(static_cast<int>(face_rect.tl_corner().x()), static_cast<int>(face_rect.tl_corner().y()) - 10);

            // 绘制背景矩形以提高文本可读性
            cv::rectangle(frame, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
            cv::putText(frame, recognized_name, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);

            // --- 最后绘制人脸矩形，以免干扰识别输入 ---
            // 显式转换为 int
            cv::rectangle(frame, cv::Point(static_cast<int>(face_rect.tl_corner().x()), static_cast<int>(face_rect.tl_corner().y())),
                          cv::Point(static_cast<int>(face_rect.br_corner().x()), static_cast<int>(face_rect.br_corner().y())), color, 2);
        }

        // 将处理后的帧发布到 MJPEG Streamer
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 80});
        streamer.publish("/webcam", std::string(buffer.begin(), buffer.end()));

        // （可选）在本地显示
        // cv::imshow("Webcam Stream", frame);
        // if (cv::waitKey(1) == 27) { // 按 ESC 退出
        //     break;
        // }
    }

    streamer.stop();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
