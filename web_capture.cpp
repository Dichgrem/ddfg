#include "config/ConfigParser.h"
#include "face/FaceRecognizer.h"
#include "timer/PerformanceTimer.h"

#include <opencv2/opencv.hpp>
#include <nadjieb/mjpeg_streamer.hpp>

#include <dlib/opencv.h>
#include <dlib/image_transforms.h>

int main() {
    ConfigParser cfg("config.yaml");
    if (!cfg.load()) return -1;

    FaceRecognizer fr("config.yaml");
    if (!fr.init()) return -1;

    PerformanceTimer& tm = PerformanceTimer::instance();

    bool useCam = cfg.get("use_camera") == "true";
    cv::VideoCapture cap;
    if (useCam) {
        cap.open(0);
    } else {
        std::string video_path = cfg.get("video_path");
        cap.open(video_path);
    }
    if (!cap.isOpened()) return -1;

    nadjieb::MJPEGStreamer streamer;
    streamer.start(8080);

    cv::Mat frame;
    while (true) {
        tm.start("帧捕获");
        cap >> frame;
        if (frame.empty()) break;
        tm.end("帧捕获");

        tm.start("人脸检测与识别");
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_BGR2RGB);
        dlib::matrix<dlib::rgb_pixel> dlib_img;
        dlib::assign_image(dlib_img, dlib::cv_image<dlib::rgb_pixel>(bgr));
        std::string name = fr.recognize(dlib_img);
        tm.end("人脸检测与识别");

        tm.start("绘制与推流");
        if (!name.empty())
            cv::putText(frame, name, {50, 50},
                        cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

        // 编码 frame 为 JPEG
        std::vector<uchar> buff;
        cv::imencode(".jpg", frame, buff);
        std::string jpeg_data(buff.begin(), buff.end());

        // 推流，路径可自定义
        streamer.publish("/stream", jpeg_data);
        tm.end("绘制与推流");
    }

    tm.printAll();
    return 0;
}

