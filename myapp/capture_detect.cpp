#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h> // 添加dlib窗口支持
#define INTV 10 //间隔多少帧做一次人脸识别
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
    
    // 创建dlib图像窗口
    dlib::image_window win;
    
    // 初始化人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
    // 主循环，持续捕获和显示视频帧
    while (true) {
        cv::Mat frame; // 创建一个Mat对象来存储当前帧
        
        // 从摄像头捕获一帧图像，运算符>>重载了视频帧捕获功能
        cap >> frame;
        
        // 检查捕获的帧是否为空,如果为空则退出循环
        if (frame.empty()) {
            cout << "没有捕获到图像，退出中..." << endl;
            break;
        }
        
        // 将OpenCV Mat转换为dlib格式
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        win.set_image(dlib_img);
        
        // 使用dlib窗口的点击事件检测代替waitKey
        if (win.is_closed()) { // 检查窗口是否被关闭
            break;
        }
        
        if (count % INTV == 0) { // 控制每INTV帧检测一次人脸
            //调用人脸检测，detector,返回人脸矩形框列表
            std::vector<dlib::rectangle> faces = detector(dlib_img);
            
            // 显示结果： 要求打印人脸数量，每个矩形框的四个点（具体上网查询）
            cout << "检测到 " << faces.size() << " 个人脸" << endl;
            for (size_t i = 0; i < faces.size(); i++) {
                cout << "人脸 #" << i << ": " 
                     << "左上角=(" << faces[i].left() << "," << faces[i].top() << "), "
                     << "右下角=(" << faces[i].right() << "," << faces[i].bottom() << ")" << endl;
            }
            
            //清空显示窗口的内容，方便重新绘制矩形框
            win.clear_overlay();
            
            // 在dlib窗口中显示图像和人脸框
            win.add_overlay(faces, dlib::rgb_pixel(255,0,0));
        }
        
        count++;
        
        // 保留一个小的延迟
        dlib::sleep(30);
    }
    
    return 0; // 程序正常退出
}