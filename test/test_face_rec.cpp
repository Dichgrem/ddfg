#include "ConfigParser.h"
#include "FaceRecognition.hpp"
#include <iostream>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

int main() {
    // 1. 加载配置
    ConfigParser config;
    if (!config.load("../config/config.json")) {
        return -1;
    }

    try {
        // 2. 初始化人脸识别模块
        FaceRecognition face_rec(config);

        // 3. 打印人脸库信息
        face_rec.printFaceLibInfo();

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        auto sp = face_rec.getShapePredictor();

        // 4. 测试一张库中存在的人脸
        std::cout << "\n--- Testing with a known face (e.g., Musk) ---" << std::endl;
        dlib::matrix<dlib::rgb_pixel> known_img;
        // 请将此路径改为 facelib 中存在的任意一张图片
        dlib::load_image(known_img, "../facelib/Elon_Musk/1.jpg"); 
        
        auto known_faces = detector(known_img);
        if (!known_faces.empty()) {
            auto shape = sp(known_img, known_faces[0]);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(known_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            
            std::string name = face_rec.recognize(face_chip);
            std::cout << "Recognition result: " << name << std::endl;
        } else {
             std::cout << "No face detected in the known face image." << std::endl;
        }


        // 5. 测试一张陌生人照片
        std::cout << "\n--- Testing with a stranger's face ---" << std::endl;
        dlib::matrix<dlib::rgb_pixel> stranger_img;
        dlib::load_image(stranger_img, "../test/stranger_face.jpg"); // 确保此测试图片存在
        
        auto stranger_faces = detector(stranger_img);
        if (!stranger_faces.empty()) {
            auto shape = sp(stranger_img, stranger_faces[0]);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(stranger_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

            std::string name = face_rec.recognize(face_chip);
            std::cout << "Recognition result: " << name << std::endl;
        } else {
            std::cout << "No face detected in the stranger's image." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
