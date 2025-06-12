#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <resnet.h>
using namespace dlib;
using namespace std;

// 定义深度神经网络模型类型
anet_type net;

// 获取目录下所有图片文件，以vector的形式返回
std::vector<string> get_image_files(const string& dir) {
    std::vector<string> files;
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) {
        cerr << "无法打开目录: " << dir << endl;
        return files;
    }
    
    while ((dirp = readdir(dp)) != NULL) {
        string filename = dirp->d_name;
        if (filename == "." || filename == "..") continue;
        
        // 检查文件扩展名
        string ext = filename.substr(filename.find_last_of(".") + 1);
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "jpg" || ext == "jpeg" || ext == "png") {
            files.push_back(filename);
        }
    }
    closedir(dp);
    return files;
}

int main() {
    // TODO: 初始化人脸检测器
    frontal_face_detector detector = get_frontal_face_detector();
    
    // TODO: 加载人脸特征点检测器(使用你自己设备上的dat文件路径)
    shape_predictor sp;
    deserialize("/home/dich/practice/model/shape_predictor_68_face_landmarks.dat") >> sp;
    
    // TODO: 加载深度神经网络模型
    deserialize("/home/dich/practice/model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    
    // TODO： 设置人脸图片目录，修改为你的实际绝对路径,注意图片要提前放入。
    string face_dir = "/home/dich/practice/dlib/myapp/facelib/face_images";
    
    // 获取所有人脸图片文件
    std::vector<string> image_files = get_image_files(face_dir);
    if (image_files.empty()) {
        cerr << "目录中没有找到图片文件" << endl;
        return -1;
    }
    
    cout << "找到 " << image_files.size() << " 张图片文件" << endl;
    
    // TODO:打开CSV文件准备写入，这里也需要搞清楚路径，自行修改
    ofstream csv_file("/home/dich/practice/dlib/myapp/facelib/facelib.csv");
    if (!csv_file.is_open()) {
        cerr << "无法创建facelib.csv文件" << endl;
        return -1;
    }
    
    int processed_count = 0;
    int total_count = image_files.size();
    
    // 处理每张图片
    for (const auto& filename : image_files) {
        // 提取人名（去掉扩展名）
        string person_name = filename.substr(0, filename.find_last_of("."));
        
        // TODO: 加载图片（filename）
        string full_path = face_dir + "/" + filename;
        matrix<rgb_pixel> img;
        try {
            load_image(img, full_path);
        } catch (const exception& e) {
            cerr << "无法加载图片: " << full_path << " - " << e.what() << endl;
            continue;
        }
        
        // TODO: 检测人脸
        std::vector<rectangle> faces = detector(img);
        if (faces.size() == 0) {
            cerr << "在图片 " << filename << " 中没有检测到人脸" << endl;
            continue;
      }
        
        if (faces.size() > 1) {
            cout << "警告: 图片 " << filename << " 中检测到多张人脸，使用第一张" << endl;
        }
        
        // TODO: 提取人脸特征点并对齐
        auto shape = sp(img, faces[0]);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        
        // TODO: 提取特征向量
        matrix<float,0,1> face_descriptor = net(face_chip);
        
        // TODO 写入CSV文件（注意路径的位置，不要放入build里面去了）
        // 先写入person_name到csv_file
        csv_file << person_name;
        
        // 在遍历face_descriptor(向量)，每次写入一个数据，并且加逗号
        for (long i = 0; i < face_descriptor.size(); ++i) {
            csv_file << "," << face_descriptor(i);
        }
        
        //一条记录完成，输入回车
        csv_file << endl;
        
        //TODO 打印提示信息，说明处理进度
        processed_count++;
        cout << "已处理: " << person_name << " (" << processed_count << "/" << total_count << ")" << endl;
    }
    
    csv_file.close();
    
    //打印提示信息，说明工作结束
    cout << "人脸库构建完成，已保存到 facelib.csv" << endl;
    cout << "总共处理了 " << processed_count << " 张人脸图片" << endl;
    
    return 0;
}
