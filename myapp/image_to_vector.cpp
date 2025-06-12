#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <iostream>
#include <resnet.h>
using namespace dlib;
using namespace std;

// 定义深度神经网络模型类型
anet_type net;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "使用: ./image_to_vector <image_path>" << std::endl;
        return -1;
    }
    
    string image_path = argv[1];
    
    // 初始化人脸检测器
    frontal_face_detector detector = get_frontal_face_detector();
    
    // 加载人脸特征点检测器(使用你自己设备上的dat文件路径)
    shape_predictor sp;
    deserialize("/home/dich/practice/model/shape_predictor_68_face_landmarks.dat") >> sp;
    
    // 加载深度神经网络模型
    deserialize("/home/dich/practice/model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    
    // 加载图像
    matrix<rgb_pixel> img;
    load_image(img, image_path);
    
    // 检测人脸,如果没有找到，退出程序
    std::vector<rectangle> faces = detector(img);
    
    if (faces.size() == 0) {
        cout << "没有检测到人脸，程序退出。" << endl;
        return -1;
    }
    
    cout << "检测到 " << faces.size() << " 张人脸，处理第一张人脸。" << endl;
    
    // 提取人脸特征点并对齐，加上上面布置提取的人脸矩形框为faces,这里只处理第一张人脸
    auto shape = sp(img, faces[0]);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
    
    // 提取特征向量
    matrix<float,0,1> face_descriptor = net(face_chip);
    
    // 打印特征向量
    cout << "对应的向量：" << endl;
    cout << "[";
    for (long i = 0; i < face_descriptor.size(); ++i) {
        cout << face_descriptor(i);
        if (i != face_descriptor.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
    
    return 0;
}
