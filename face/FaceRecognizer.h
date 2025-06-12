#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <string>
#include <memory>
#include <dlib/matrix.h>
#include <dlib/pixel.h>

class FaceRecognizer {
public:
    explicit FaceRecognizer(const std::string& configPath);
    ~FaceRecognizer();

    bool init();
    void loadLibrary(const std::string& libCsv);
    std::string recognize(const dlib::matrix<dlib::rgb_pixel>& face);
    void printLibrary() const;

private:
    struct Impl; // 声明内部结构体
    std::unique_ptr<Impl> impl_; // 实现细节成员变量
};

#endif // FACE_RECOGNIZER_H

