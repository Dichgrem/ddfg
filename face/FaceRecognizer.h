#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <memory>
#include <string>
#include <dlib/matrix.h>

class FaceRecognizer {
public:
    explicit FaceRecognizer(const std::string& configPath);
    ~FaceRecognizer();

    bool init();
    std::string recognize(const dlib::matrix<dlib::rgb_pixel>& img);
    void printLibrary() const;

private:
    struct Impl;              // 仅前向声明
    Impl* impl_;              // 用裸指针，或者
    // std::unique_ptr<Impl> impl_;  // 也可以用unique_ptr，但要自己写构造/析构
};

#endif

