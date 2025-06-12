#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <string>
#include <vector>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>

using namespace dlib;

class FaceRecognizer {
public:
    explicit FaceRecognizer(const std::string& configPath);
    bool init();
    void loadLibrary(const std::string& libCsv);
    std::string recognize(const matrix<rgb_pixel>& face);
    void printLibrary() const;

private:
    std::string _config;
    double _threshold{0};
    shape_predictor _shapePredictor;
    anet_type _net;
    std::vector<std::string> _names;
    std::vector<matrix<float,0,1>> _embeddings;
};

#endif // FACE_RECOGNIZER_H

