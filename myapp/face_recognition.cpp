#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <resnet.h>
using namespace dlib;
using namespace std;

// 定义深度神经网络模型类型
anet_type net;

// 人脸数据库条目结构
struct FaceRecord {
    string name; //姓名
    matrix<float,0,1> descriptor; //128维度人脸向量
};

// 加载人脸数据库
std::vector<FaceRecord> load_face_lib(const string& filename) {
    std::vector<FaceRecord> face_lib;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        istringstream iss(line);
        FaceRecord record;
        string value;
        
        // 读取人名
        getline(iss, record.name, ',');
        
        // 读取特征向量
        matrix<float,0,1> descriptor(128);
        for (long i = 0; i < 128; ++i) {
            getline(iss, value, ',');
            descriptor(i) = stof(value);
        }
        record.descriptor = descriptor;
        face_lib.push_back(record);
    }
    
    cout << "加载了 " << face_lib.size() << " 个人脸记录到数据库" << endl;
    return face_lib;
}

// 识别单个人脸
string recognize_face(const matrix<float,0,1>& descriptor, //当前的人脸向量
                     const std::vector<FaceRecord>& face_lib, //人脸数据库装载到vector中
                     double threshold = 0.6) {
    string best_match = "未知";
    double min_distance = 1.0; // 设置一个足够大的初始值
    
    // TODO： 遍历face_lib，逐个对比，判断descriptor和face_lib的哪一个成员最接近，将人名记录到best_match中
    for (const auto& record : face_lib) {
        // 计算欧氏距离
        double distance = length(descriptor - record.descriptor);
        
        // 如果距离更小，更新最佳匹配
        if (distance < min_distance) {
            min_distance = distance;
            best_match = record.name;
        }
    }
    
    // 如果最小距离超过阈值，认为是未知人员
    if (min_distance > threshold) {
        best_match = "未知";
    }
    
    cout << "  最佳匹配距离: " << min_distance << " (阈值: " << threshold << ")" << endl;
    
    return best_match;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "用法: ./face_recognition <合照图片路径>" << endl;
        return -1;
    }
    
    // 加载人脸数据库
    std::vector<FaceRecord> face_lib = load_face_lib("facelib.csv");
    if (face_lib.empty()) {
        cerr << "人脸数据库加载失败或为空" << endl;
        return -1;
    }
    
    // 初始化人脸检测器
    frontal_face_detector detector = get_frontal_face_detector();
    
    // 加载人脸特征点检测器
    shape_predictor sp;
    deserialize("/home/dich/practice/model/shape_predictor_68_face_landmarks.dat") >> sp;
    
    // 加载深度神经网络模型
    deserialize("/home/dich/practice/model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    
    // 加载合照图片
    matrix<rgb_pixel> img;
    try {
        load_image(img, argv[1]);
    } catch (const exception& e) {
        cerr << "无法加载图片: " << argv[1] << " - " << e.what() << endl;
        return -1;
    }
    
    // 检测合照中所有人脸，使用detector(img)获取所有人脸矩形vector
    std::vector<rectangle> faces = detector(img);
    
    //提示信息，说明识别到多少张人脸
    if (faces.size() == 0) {
        cout << "没有检测到人脸" << endl;
        return -1;
    }
    
    cout << "检测到 " << faces.size() << " 张人脸，开始识别..." << endl;
    
    // 处理每张人脸
    for (size_t i = 0; i < faces.size(); ++i) {
        cout << "正在处理人脸 " << (i + 1) << ":" << endl;
        
        // 提取人脸特征点并对齐
        auto shape = sp(img, faces[i]);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        
        // 提取特征向量
        matrix<float,0,1> face_descriptor = net(face_chip);
        
        // 调用recognize_face识别人脸
        // 可以根据测试结果调整阈值，比如：
        // - 0.6: 较严格，减少误识别但可能增加"未知"
        // - 0.7: 中等严格程度
        // - 0.8: 较宽松，可能增加误识别但减少"未知"
        string identity = recognize_face(face_descriptor, face_lib, 0.6);
        
        // 输出识别结果
        cout << "人脸 " << (i + 1) << ": " << identity << endl;
        cout << endl;
    }
    
    return 0;
}
