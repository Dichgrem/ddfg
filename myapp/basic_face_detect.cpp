#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>  // 添加显示窗口支持

using namespace std;

int main(int argc, char** argv) {
    
    // 检查命令行参数
    if (argc != 2) {
        cerr << "使用方法: " << argv[0] << " <图片路径>" << endl;
        return 1;
    }

    try {
        // 加载图片
        dlib::array2d<dlib::rgb_pixel> img;
        load_image(img, argv[1]);  
    
        // 初始化人脸检测器
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
        // 检测人脸：调用人脸检测detector,返回人脸矩形框列表
        std::vector<dlib::rectangle> faces = detector(img);

        // 显示结果
        cout << "检测到 " << faces.size() << " 张人脸" << endl;
        
        // 可视化结果
        dlib::image_window win;
        win.set_image(img);
        win.add_overlay(faces, dlib::rgb_pixel(255,0,0));
        
        cout << "按回车键退出..." << endl;
        cin.get();
        
    } catch (exception& e) {
        cout << "发生异常: " << e.what() << endl;
    }
    
    return 0;
}