#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>  // 添加dlib窗口支持

using namespace std;

int main() {
    // 创建VideoCapture对象，参数0表示打开默认摄像头，也可以指定视频文件路径或摄像头索引号
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    // 创建dlib图像窗口
    dlib::image_window win;
    
    while (!win.is_closed()) {
        cv::Mat frame;
        //从摄像头捕获一帧图像，运算符>>重载了视频帧捕获功能
        cap >> frame;
        
        if (frame.empty()) break;
        
        // 将OpenCV Mat转换为dlib格式,存储图片的变量为dlib_img
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        
        // 在dlib窗口中显示图像dlib_img
        win.set_image(dlib_img);
        
        // 延迟30ms再采集下一帧
        dlib::sleep(30);
    }
    cap.release();
    return 0;
}
