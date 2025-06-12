#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <nadjieb/mjpeg_streamer.hpp> 
#include <filesystem>
#include <chrono> //计时用

#define INTV 10 //间隔10帧做一次人脸识别
#define FACE_SAVE_INTERVAL 2 // 每隔2秒保存一次对齐后的人脸

using MJPEGStreamer = nadjieb::MJPEGStreamer;
using namespace std; 
namespace fs = std::filesystem;

int main() {
    int count = 0;
    
    // 记录上次保存人脸的时间
    auto last_save_time = std::chrono::steady_clock::now();
    
    // 创建保存对齐人脸的目录
    const string face_dir = "./aligned_faces";
    if (!fs::exists(face_dir)) {
        fs::create_directory(face_dir);
    }
    
    // 打开默认摄像头
    cv::VideoCapture cap(0);
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        cout << "无法打开摄像头!" << endl;
        return -1;
    }
    
    // 初始化人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
    // 加载模型
    dlib::shape_predictor sp;
    try {
        dlib::deserialize("/home/dich/practice/model/shape_predictor_68_face_landmarks.dat") >> sp;
        cout << "成功加载人脸特征点检测模型" << endl;
    } catch (const std::exception& e) {
        cout << "无法加载人脸特征点检测模型: " << e.what() << endl;
        cout << "确保模型文件路径正确" << endl;
        return -1;
    }
    
    // 初始化MJPEG流服务器
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    MJPEGStreamer streamer;
    streamer.start(8080);
    
    cout << "MJPEG流已启动，请访问 http://localhost:8080/video 查看" << endl;
    
    // 主循环，持续捕获和显示视频帧
    while (streamer.isRunning()) {
        cv::Mat frame; // 创建一个Mat对象来存储当前帧
        
        // 从摄像头捕获一帧图像，运算符>>重载了视频帧捕获功能
        cap >> frame;
        
        // 检查捕获的帧是否为空,如果为空则退出循环
        if (frame.empty()) {
            cout << "没有捕获到图像，退出中..." << endl;
            break;
        }
        
        // 将OpenCV Mat转换为dlib格式进行人脸检测
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        
        if (count % INTV == 0) { // 控制每INTV帧检测一次人脸
            //调用人脸检测，detector,返回人脸矩形框列表
            std::vector<dlib::rectangle> faces = detector(dlib_img);
            
            // 显示结果：打印人脸数量和每个矩形框的位置
            cout << "检测到 " << faces.size() << " 个人脸" << endl;
            
            // 计算自上次保存以来经过的时间（秒）
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - last_save_time).count();
            
            // 判断是否达到保存间隔
            bool should_save = elapsed_seconds >= FACE_SAVE_INTERVAL;
            
            // 在frame上绘制人脸矩形框并对齐保存
            for (size_t i = 0; i < faces.size(); i++) {
                cout << "人脸 #" << i << ": " 
                     << "左上角=(" << faces[i].left() << "," << faces[i].top() << "), "
                     << "右下角=(" << faces[i].right() << "," << faces[i].bottom() << ")" << endl;
                
                // 在OpenCV frame上绘制矩形框
                cv::rectangle(frame, 
                              cv::Point(faces[i].left(), faces[i].top()),
                              cv::Point(faces[i].right(), faces[i].bottom()),
                              cv::Scalar(0, 255, 0), 2);
                
                // 每隔 FACE_SAVE_INTERVAL 秒保存一次对齐后的人脸
                if (should_save) {
                    try {
                        // 提取人脸特征点
                        dlib::full_object_detection shape = sp(dlib_img, faces[i]);
                        
                        // 创建对齐的人脸图像
                        dlib::matrix<dlib::rgb_pixel> face_chip;
                        dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                        
                        // 生成唯一的文件名（使用时间戳）
                        string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                        string face_filename = face_dir + "/face_" + timestamp + "_" + to_string(i) + ".jpg";
                        
                        // 保存对齐后的人脸图像
                        dlib::save_jpeg(face_chip, face_filename);
                        cout << "保存对齐人脸: " << face_filename << endl;
                    } catch (const std::exception& e) {
                        cout << "对齐人脸失败: " << e.what() << endl;
                    }
                }
            }
            
            // 如果保存了人脸，更新上次保存时间
            if (should_save && !faces.empty()) {
                last_save_time = current_time;
                cout << "----------- 已保存人脸，下次保存将在 " << FACE_SAVE_INTERVAL << " 秒后 -----------" << endl;
            }
        }
        
        count++;
        
        // 将当前帧转为jpg并发布到流
        // http://localhost:8080/video
        std::vector<uchar> buff_video;
        cv::imencode(".jpg", frame, buff_video, params);
        streamer.publish("/video", std::string(buff_video.begin(), buff_video.end()));
        
        // 短暂休眠以控制帧率
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    // 停止流服务器
    streamer.stop();
    
    return 0; // 程序正常退出
}