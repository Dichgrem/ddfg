#ifndef FACERECOGNITION_HPP
#define FACERECOGNITION_HPP

#include "ConfigParser.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace dr = dlib;

/// 与 dnn_face_recognition_ex.cpp 中定义相同的网络类型
template <template<int,template<class>class,int,typename> class BLOCK,
          int N, template<class>class BN, int stride, typename SUBNET>
using residual = dr::add_prev1<BLOCK<N,BN,stride,dr::tag1<SUBNET>>>;

template <template<int,template<class>class,int,typename> class BLOCK,
          int N, template<class>class BN, int stride, typename SUBNET>
using residual_down =
    dr::add_prev2<
      dr::avg_pool<2,2,2,2,
        dr::skip1<BLOCK<N,BN,stride,dr::tag1<SUBNET>>>
      >
    >;

template <int N, template<class>class BN, int stride, typename SUBNET>
using block  = BN<
                   dr::con<N,3,3,1,1,
                     dr::relu<
                       BN<dr::con<N,3,3,stride,stride,SUBNET>>
                     >
                   >
                 >;

template <int N, typename SUBNET>
using res      = dr::relu< residual<block,N,dr::bn_con,1,SUBNET> >;

template <int N, typename SUBNET>
using res_down = dr::relu< residual_down<block,N,dr::bn_con,2,SUBNET> >;

/// 最终网络：150×150 输入 -> 128d 向量
using anet_type = dr::loss_metric<
                     dr::fc_no_bias<128,
                     dr::avg_pool_everything<
                       res<256,
                       res<256,
                       res_down<256,
                       res<128,
                       res<128,
                       res_down<128,
                       res<64,
                       res<64,
                       res<64,
                       res_down<64,
                       res<32,
                       dr::input_rgb_image_sized<150>
                       >>>>>>>>>>>>>>;

class FaceRecognition {
public:
    /// 构造：加载配置、模型，并根据配置加载人脸库
    explicit FaceRecognition(const ConfigParser& config);

    /// 识别已对齐的 150×150 人脸图，返回姓名或 "Stranger"
    std::string recognize(const dr::matrix<dr::rgb_pixel>& face_chip);

    /// 打印当前人脸库信息（条目数、阈值）
    void printFaceLibInfo() const;

    /// 获取内部 shape_predictor（用于外部提取人脸关键点）
    dr::shape_predictor getShapePredictor() const;

private:
    void loadModels(const ConfigParser& config);
    void loadLibraryFromCSV(const std::string& csv_path);
    void buildFaceLibrary(const std::string& dir_path);

    dr::shape_predictor                          sp_;
    anet_type                                    net_;
    double                                       face_match_threshold_;
    std::map<std::string, dr::matrix<float,0,1>> face_library_;
};

#endif // FACERECOGNITION_HPP

