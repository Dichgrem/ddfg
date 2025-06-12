#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <nadjieb/mjpeg_streamer.hpp> // 添加视频流相关头文件

#define INTV 10 //间隔多少帧做一次人脸识别

// for convenience
using MJPEGStreamer = nadjieb::MJPEGStreamer;
using namespace std; // 使用标准命名空间

int main() {
    int count = 0;
    // 创建VideoCapture对象，参数0表示打开默认摄像头，也可以指定视频文件路径或摄像头索引号
    cv::VideoCapture cap(0);
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        cout << "无法打开摄像头!" << endl;
        return -1;
    }
    
    // 初始化人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
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
            
            // 在frame上绘制人脸矩形框
            for (size_t i = 0; i < faces.size(); i++) {
                cout << "人脸 #" << i << ": " 
                     << "左上角=(" << faces[i].left() << "," << faces[i].top() << "), "
                     << "右下角=(" << faces[i].right() << "," << faces[i].bottom() << ")" << endl;
                
                // 在OpenCV frame上绘制矩形框
                cv::rectangle(frame, 
                              cv::Point(faces[i].left(), faces[i].top()),
                              cv::Point(faces[i].right(), faces[i].bottom()),
                              cv::Scalar(0, 255, 0), 2);
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