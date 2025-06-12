#include "ConfigParser.h"
#include "FaceRecognition.h"
#include "PerformanceTimer.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>
#include <nadjieb/mjpeg_streamer.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <csignal>

// 使用dlib的图像类型，避免和OpenCV的混淆
using dlib::cv_image;
using dlib::rgb_pixel;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

// 全局变量用于信号处理
volatile sig_atomic_t g_signal_status = 0;

void signal_handler(int signal) {
    g_signal_status = signal;
}


int main() {
    // 注册信号处理器，用于优雅地退出并打印性能报告
    std::signal(SIGINT, signal_handler); // Ctrl+C

    // 1. 初始化模块
    auto& timer = TimerManager::getInstance();
    ConfigParser config;
    if (!config.load("./config/config.json")) {
        return -1;
    }
    
    std::unique_ptr<FaceRecognition> face_rec;
    try {
        face_rec = std::make_unique<FaceRecognition>(config);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize FaceRecognition: " << e.what() << std::endl;
        return -1;
    }
    face_rec->printFaceLibInfo();

    // 2. 配置视频源
    cv::VideoCapture cap;
    if (config.get<bool>("use_camera", true)) {
        cap.open(0);
    } else {
        cap.open(config.get<std::string>("video_path", ""));
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video source." << std::endl;
        return -1;
    }

    // 3. 初始化dlib人脸检测器和对齐器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp = face_rec->getShapePredictor();

    // 4. 初始化MJPEG推流服务
    MJPEGStreamer streamer;
    streamer.start(8080);
    std::cout << "\nMJPEG Streamer started on port 8080." << std::endl;
    std::cout << "Open http://localhost:8080 in your browser." << std::endl;
    
    // 获取配置
    int frame_interval = config.get<int>("frame_sample_interval", 1);
    int frame_count = 0;

    // 注册退出时打印性能报告
    std::atexit([](){ TimerManager::getInstance().printAllSummaries(); });

    // 5. 主循环
    while (streamer.isAlive() && g_signal_status == 0) {
        timer.start("Frame Total");
        
        cv::Mat temp;
        timer.start("Frame Capture");
        if (!cap.read(temp)) {
            std::cout << "Video ended or failed to capture frame." << std::endl;
            break;
        }
        timer.end("Frame Capture");

        // 转换为dlib格式
        cv_image<rgb_pixel> img(temp);

        // 每隔N帧检测一次，以提高性能
        if (frame_count % frame_interval == 0) {
            timer.start("Face Detection");
            std::vector<dlib::rectangle> dets = detector(img);
            timer.end("Face Detection");

            // 存储检测结果，在每一帧都绘制
            std::vector<std::pair<dlib::rectangle, std::string>> face_results;
            
            for (auto&& d : dets) {
                timer.start("Face Alignment");
                auto shape = sp(img, d);
                dlib::matrix<rgb_pixel> face_chip;
                dlib::extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                timer.end("Face Alignment");
                
                timer.start("Face Recognition");
                std::string name = face_rec->recognize(face_chip);
                timer.end("Face Recognition");
                
                face_results.push_back({d, name});
            }

            // 绘制矩形和姓名
            for (const auto& res : face_results) {
                 cv::Rect r(res.first.left(), res.first.top(), res.first.width(), res.first.height());
                 cv::Scalar color = (res.second == "Stranger") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                 cv::rectangle(temp, r, color, 2);

                 cv::Point text_pos(r.x, r.y - 10 > 0 ? r.y - 10 : r.y + 10);
                 cv::putText(temp, res.second, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
        }
        
        frame_count++;
        
        timer.start("MJPEG Streaming");
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", temp, buff_bgr);
        streamer.publish("/bgr", std::string(buff_bgr.begin(), buff_bgr.end()));
        timer.end("MJPEG Streaming");

        timer.end("Frame Total");
    }

    streamer.stop();
    std::cout << "Streamer stopped. Exiting." << std::endl;
    // atexit 会在这里被调用，打印性能报告

    return 0;
}
