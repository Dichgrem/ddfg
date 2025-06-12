#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include "ConfigParser.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <map>
#include <string>
#include <vector>

// 定义残差块和网络架构
template <template <int, template<typename>class, int, typename> class BLOCK, int N, template<typename>class BN, int stride, typename SUBNET>
using residual = dlib::add_prev1<dlib::relu<BN<dlib::con<N, 3, 3, 1, 1, BLOCK<N, BN, stride, SUBNET>>>>>;

template <template <int, template<typename>class, int, typename> class BLOCK, int N, template<typename>class BN, int stride, typename SUBNET>
using residual_down = dlib::add_prev1<dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, BLOCK<N, BN, stride, SUBNET>>>>>;

template <int N, template<typename>class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::bn_con, 1, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::bn_con, 2, SUBNET>>;

// 定义 anet_type，使用 bn_con 进行批归一化
using anet_type = dlib::loss_metric<
    dlib::fc_no_bias<128,
    dlib::avg_pool_everything<
        ares<256,
        ares<256,
        ares_down<256,
        ares<128,
        ares<128,
        ares_down<128,
        ares<64,
        ares<64,
        ares<64,
        ares_down<64,
        ares<32,
        dlib::input<dlib::matrix<dlib::rgb_pixel>>>>>>>>>>>>>;

class FaceRecognition {
public:
    explicit FaceRecognition(const ConfigParser& config);
    std::string recognize(const dlib::matrix<dlib::rgb_pixel>& face_chip);
    void printFaceLibInfo() const;
    dlib::shape_predictor getShapePredictor() const;

private:
    void loadModels(const ConfigParser& config);
    void buildFaceLibrary(const std::string& lib_path);

    dlib::shape_predictor sp_;
    anet_type net_;
    double face_match_threshold_;
    std::map<std::string, dlib::matrix<float, 0, 1>> face_library_;
};

#endif // FACE_RECOGNITION_H
