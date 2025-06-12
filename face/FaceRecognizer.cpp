#include "FaceRecognizer.h"
#include "../config/ConfigParser.h"
#include <dlib/image_io.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

FaceRecognizer::FaceRecognizer(const std::string& configPath)
    : _config(configPath) {}

bool FaceRecognizer::init() {
    ConfigParser cfg(_config);
    if (!cfg.load()) return false;

    _threshold = std::stod(cfg.get("face_threshold"));
    deserialize(cfg.get("shape_model")) >> _shapePredictor;
    deserialize(cfg.get("dnn_model")) >> _net;

    std::string libCsv = cfg.get("face_lib_csv");
    if (!libCsv.empty()) loadLibrary(libCsv);

    return true;
}

void FaceRecognizer::loadLibrary(const std::string& libCsv) {
    std::ifstream fin(libCsv);
    if (!fin) {
        std::cerr << "Cannot open face library: " << libCsv << std::endl;
        return;
    }
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::string name;
        iss >> name;
        matrix<float,0,1> emb;
        emb.set_size(128);
        for (int i = 0; i < 128; ++i) iss >> emb(i);
        _names.push_back(name);
        _embeddings.push_back(emb);
    }
}

std::string FaceRecognizer::recognize(const matrix<rgb_pixel>& face) {
    matrix<rgb_pixel> aligned;
    auto dets = get_frontal_face_detector()(face);
    if (dets.empty()) return "";

    auto shape = _shapePredictor(face, dets[0]);
    extract_image_chip(face, get_face_chip_details(shape,150,0.25), aligned);

    auto faceDesc = _net(aligned);

    double best = 1e9;
    int bestIdx = -1;
    for (size_t i = 0; i < _embeddings.size(); ++i) {
        auto dist = length(_embeddings[i] - faceDesc);
        if (dist < best) {
            best = dist;
            bestIdx = i;
        }
    }
    if (bestIdx >= 0 && best < _threshold)
        return _names[bestIdx];
    return "";
}

void FaceRecognizer::printLibrary() const {
    std::cout << "人脸库信息:" << std::endl;
    for (auto& n : _names) {
        std::cout << "  " << n << std::endl;
    }
}

