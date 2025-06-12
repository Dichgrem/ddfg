#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <iostream>
#include <resnet.h>
using namespace dlib;
using namespace std;

// 定义深度神经网络模型类型
anet_type net;

// 计算余弦相似度的函数
double cosine_similarity(const matrix<float,0,1>& A, const matrix<float,0,1>& B) {
    double dot_product = dot(A, B);
    double norm_a = length(A);
    double norm_b = length(B);
    return dot_product / (norm_a * norm_b);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "用法: ./face_cmp <图片路径1> <图片路径2>" << std::endl;
        return -1;
    }
    
    // TODO: 初始化人脸检测器
    frontal_face_detector detector = get_frontal_face_detector();
    
    // TODO: 加载人脸特征点检测器(使用你自己设备上的dat文件路径)
    shape_predictor sp;
    deserialize("/home/dich/practice/model/shape_predictor_68_face_landmarks.dat") >> sp;
    
    // TODO: 加载深度神经网络模型
    deserialize("/home/dich/practice/model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    
    // TODO: 从argv[1]加载第一张图像
    matrix<rgb_pixel> img1;
    load_image(img1, argv[1]);
    
    // TODO: 从argv[2]加载第二张图像
    matrix<rgb_pixel> img2;
    load_image(img2, argv[2]);
    
    // TODO: 检测第一张图像中的人脸faces1,如果没有人脸就退出
    std::vector<rectangle> faces1 = detector(img1);
    if (faces1.size() == 0) {
        cout << "第一张图片中没有检测到人脸，程序退出。" << endl;
        return -1;
    }
    
    // TODO: 检测第二张图像中的人脸faces2,如果没有人脸就退出
    std::vector<rectangle> faces2 = detector(img2);
    if (faces2.size() == 0) {
        cout << "第二张图片中没有检测到人脸，程序退出。" << endl;
        return -1;
    }
    
    cout << "第一张图片检测到 " << faces1.size() << " 张人脸" << endl;
    cout << "第二张图片检测到 " << faces2.size() << " 张人脸" << endl;
    
    // TODO：提取第一张图像的特征点（调用sp）并对齐extract_image_chip
    auto shape1 = sp(img1, faces1[0]);
    matrix<rgb_pixel> face_chip1;
    extract_image_chip(img1, get_face_chip_details(shape1, 150, 0.25), face_chip1);
    
    // TODO：提取第二张图像的特征点（调用sp）并对齐extract_image_chip
    auto shape2 = sp(img2, faces2[0]);
    matrix<rgb_pixel> face_chip2;
    extract_image_chip(img2, get_face_chip_details(shape2, 150, 0.25), face_chip2);
    
    // 分别提取两张图片的特征向量face_descriptor1， face_descriptor2
    matrix<float,0,1> face_descriptor1 = net(face_chip1);
    matrix<float,0,1> face_descriptor2 = net(face_chip2);
    
    // 计算两个特征向量之间的余弦相似度
    double cos_sim = cosine_similarity(face_descriptor1, face_descriptor2);
    
    // 计算两个特征向量之间的欧氏距离
    double distance = length(face_descriptor1 - face_descriptor2);
    
    // 将distance和cos_sim打印出来
    cout << "人脸欧氏距离: " << distance << endl;
    cout << "余弦相似度: " << cos_sim << endl;
    
    //TODO: 根据情况调整这个阈值，请根据实际情况调整。打印信息判断是否是同一个人。
    // 根据经验值设置阈值：
    // 欧氏距离 < 0.6 且余弦相似度 > 0.85 认为是同一个人
    if (distance < 0.6 && cos_sim > 0.85) {
        cout << "人脸相似度高，很可能是同一个人" << endl;
    } else if (distance < 0.8 && cos_sim > 0.75) {
        cout << "人脸有一定相似度，可能是同一个人" << endl;
    } else {
        cout << "人脸相似度低，很可能不是同一个人" << endl;
    }
    
    return 0;
}
