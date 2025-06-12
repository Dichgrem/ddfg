#include "FaceRecognition.h"
#include <iostream>
#include <filesystem>

FaceRecognition::FaceRecognition(const ConfigParser& config) {
    std::cout << "Initializing FaceRecognition module..." << std::endl;
    loadModels(config);
    buildFaceLibrary(config.get<std::string>("face_library_path", ""));
    face_match_threshold_ = config.get<double>("face_match_threshold", 0.6);
}

void FaceRecognition::loadModels(const ConfigParser& config) {
    auto sp_path = config.get<std::string>("models.shape_predictor", "");
    auto net_path = config.get<std::string>("models.face_recognition", "");

    if (sp_path.empty() || net_path.empty()){
        throw std::runtime_error("Model paths are not configured in config file.");
    }
    
    try {
        dlib::deserialize(sp_path) >> sp_;
        dlib::deserialize(net_path) >> net_;
        std::cout << "Dlib models loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading models: " << e.what() << std::endl;
        throw;
    }
}

void FaceRecognition::buildFaceLibrary(const std::string& lib_path) {
    if (lib_path.empty() || !std::filesystem::exists(lib_path)) {
        std::cerr << "Face library path does not exist: " << lib_path << std::endl;
        return;
    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    for (const auto& entry : std::filesystem::directory_iterator(lib_path)) {
        if (entry.is_directory()) {
            std::string person_name = entry.path().filename().string();
            std::cout << "Processing person: " << person_name << std::endl;

            for (const auto& img_entry : std::filesystem::directory_iterator(entry.path())) {
                dlib::matrix<dlib::rgb_pixel> img;
                try {
                    dlib::load_image(img, img_entry.path().string());
                    
                    auto faces = detector(img);
                    if (faces.size() != 1) {
                         std::cerr << "Warning: Skipping image " << img_entry.path().string() 
                                   << " for " << person_name << ". Found " << faces.size() << " faces, expected 1." << std::endl;
                        continue;
                    }

                    auto shape = sp_(img, faces[0]);
                    dlib::matrix<dlib::rgb_pixel> face_chip;
                    dlib::extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                    
                    dlib::matrix<float, 0, 1> face_descriptor = net_(face_chip);
                    face_library_[person_name] = face_descriptor;
                    
                    // 只处理每个文件夹的第一张有效图片
                    break; 

                } catch(const std::exception& e) {
                    std::cerr << "Error processing image " << img_entry.path().string() << ": " << e.what() << std::endl;
                }
            }
        }
    }
    std::cout << "Face library built. Total unique faces: " << face_library_.size() << std::endl;
}

std::string FaceRecognition::recognize(const dlib::matrix<dlib::rgb_pixel>& face_chip) {
    if (face_library_.empty()) {
        return "Stranger";
    }

    dlib::matrix<float, 0, 1> face_descriptor = net_(face_chip);
    
    double min_dist = 1.0;
    std::string matched_name = "Stranger";

    for (const auto& pair : face_library_) {
        double dist = dlib::length(pair.second - face_descriptor);
        if (dist < min_dist) {
            min_dist = dist;
            matched_name = pair.first;
        }
    }

    if (min_dist <= face_match_threshold_) {
        return matched_name;
    }

    return "Stranger";
}

void FaceRecognition::printFaceLibInfo() const {
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Face Library Information" << std::endl;
    std::cout << "Loaded faces count: " << face_library_.size() << std::endl;
    std::cout << "Matching threshold: " << face_match_threshold_ << std::endl;
    for (const auto& pair : face_library_) {
        std::cout << " - Name: " << pair.first << ", Feature dimension: " << pair.second.size() << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;
}

dlib::shape_predictor FaceRecognition::getShapePredictor() const {
    return sp_;
}
