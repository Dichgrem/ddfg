#include "../face/FaceRecognizer.h"
#include "../timer/PerformanceTimer.h"
#include <dlib/image_io.h>
#include <iostream>

int main() {
    FaceRecognizer fr("../config.yaml");
    if (!fr.init()) return -1;
    fr.printLibrary();

    matrix<rgb_pixel> img;
    load_image(img, "test_images/musk_face.jpg");
    std::string name = fr.recognize(img);
    std::cout << "识别结果: " << (name.empty()?"陌生人":name) << std::endl;

    load_image(img, "test_images/unknown.jpg");
    name = fr.recognize(img);
    std::cout << "识别结果: " << (name.empty()?"陌生人":name) << std::endl;
    return 0;
}

