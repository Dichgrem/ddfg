#include "face/FaceRecognizer.h"
#include <dlib/image_io.h>  // for load_image
#include <iostream>

int main() {
    FaceRecognizer recognizer("config/face.cfg");
    if (!recognizer.init()) {
        std::cerr << "Init failed!" << std::endl;
        return 1;
    }

    dlib::matrix<dlib::rgb_pixel> img;
    dlib::load_image(img, "test_images/musk_face.jpg");

    std::string name = recognizer.recognize(img);
    std::cout << "识别结果: " << name << std::endl;

    return 0;
}

