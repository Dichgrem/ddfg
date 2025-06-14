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
#include <csignal> // For signal handling

// 项目自定义头文件
#include "ConfigParser.h"    // 位于 include/
#include "FaceRecognition.hpp" // 位于 include/
#include "PerformanceMonitor.h" // <-- 添加这一行

// MJPEG Streamer 的头文件路径
#include <nadjieb/mjpeg_streamer.hpp> // 确保这个路径和文件存在

namespace fs = std::filesystem; // 使用 std::filesystem 命名空间

// 定义一个信号处理函数，以便在程序退出时打印报告
void signalHandler(int signum) {
    std::cout << "\n收到中断信号 (" << signum << ")。\n";
    PerformanceMonitor::getInstance().printReport(); // 打印性能报告
    exit(signum); // 正常退出
}

int main() {
    // 注册信号处理函数，以便在 Ctrl+C 退出时打印报告
    std::signal(SIGINT, signalHandler); // 处理 Ctrl+C

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
    nadjieb::MJPEGStreamer streamer;
    streamer.start(8080); // 在 8080 端口启动流

    // Dlib 人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    cv::Mat frame;
    long long frame_counter = 0; // 用于统计处理了多少帧
    const int REPORT_INTERVAL_FRAMES = 5; // <-- 调整为5帧，方便调试时快速看到报告

    while (true) {
        // 统计一帧的总处理时间
        PM_START("总帧处理");
        PerformanceMonitor::getInstance().startFrame(); // 统计总帧率

        cap >> frame;
        if (frame.empty()) {
            std::cerr << "帧为空。退出程序。" << std::endl;
            break;
        }

        // --- 人脸检测 ---
        PM_SCOPED(人脸检测); // 使用 RAII 宏，自动开始和结束计时
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        // --- 人脸处理与识别 ---
        PM_START("人脸处理与识别（总）"); // 手动开始/停止
        for (const auto& face_rect : faces) {
            PM_SCOPED(形状预测);
            dlib::full_object_detection shape = face_recognizer.getShapePredictor()(dlib_img, face_rect);

            PM_SCOPED(人脸芯片提取);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

            PM_SCOPED(核心人脸识别);
            std::string recognized_name = face_recognizer.recognize(face_chip);

            // --- 在图像上绘制人脸信息 ---
            PM_SCOPED(绘制覆盖物);
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
        PM_STOP("人脸处理与识别（总）"); // 停止手动开始的任务

        // --- 图像编码和发布 ---
        PM_SCOPED(图像编码与发布);
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 80});
        streamer.publish("/webcam", std::string(buffer.begin(), buffer.end()));

        PerformanceMonitor::getInstance().stopFrame(); // 停止总帧率统计
        PM_STOP("总帧处理"); // 停止一帧的总处理时间

        frame_counter++;
        if (frame_counter % REPORT_INTERVAL_FRAMES == 0) {
            std::cout << "\n已处理 " << frame_counter << " 帧。生成报告...\n";
            PerformanceMonitor::getInstance().printReport();
            // PerformanceMonitor::getInstance().reset(); // 如果需要，可以重置统计数据
        }

        // （可选）在本地显示
        // cv::imshow("Webcam Stream", frame); // <--- 重新注释掉此行
        // if (cv::waitKey(1) == 27) { // 按 ESC 退出
        //     break;
        // }
    }

    streamer.stop();
    cap.release();
    cv::destroyAllWindows(); // <-- 尽管不再显示窗口，但保留此行通常无害

    // 在程序正常退出前打印最终报告
    PerformanceMonitor::getInstance().printReport();

    return 0;
}
