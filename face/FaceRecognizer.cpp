// face/FaceRecognizer.cpp

#include "FaceRecognizer.h"
#include "config/ConfigParser.h"

#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

namespace {

using namespace dlib;
// ---------------------------------------------
// 1) 定义残差网络结构（摘自 dlib 官方示例）
// ---------------------------------------------
template <
    int N,
    template <typename> class BN,
    int stride,
    typename SUBNET
>
using block = BN<
                con<N,3,3,1,1,
                  relu<
                    BN<con<N,3,3,stride,stride,SUBNET>>
                  >
                >
              >;

template <
    template <int, template <typename> class, int, typename> class BLOCK,
    int N,
    template <typename> class BN,
    typename SUBNET
>
using residual = add_prev1<BLOCK<N,BN,1, tag1<SUBNET>>>;

template <
    template <int, template <typename> class, int, typename> class BLOCK,
    int N,
    template <typename> class BN,
    typename SUBNET
>
using residual_down = add_prev2<
                        avg_pool<2,2,2,2,
                          skip1<BLOCK<N,BN,2, tag2<SUBNET>>>
                        >
                      >;

template <typename SUBNET> using level1  = relu< residual<block,256,affine,SUBNET> >;
template <typename SUBNET> using level2  = relu< residual<block,256,affine,level1<SUBNET>> >;
template <typename SUBNET> using level3  = relu< residual_down<block,256,affine,level2<SUBNET>> >;
template <typename SUBNET> using level4  = relu< residual<block,128,affine,level3<SUBNET>> >;
template <typename SUBNET> using level5  = relu< residual<block,128,affine,level4<SUBNET>> >;
template <typename SUBNET> using level6  = relu< residual_down<block,128,affine,level5<SUBNET>> >;
template <typename SUBNET> using level7  = relu< residual<block,64,affine,level6<SUBNET>> >;
template <typename SUBNET> using level8  = relu< residual<block,64,affine,level7<SUBNET>> >;
template <typename SUBNET> using level9  = relu< residual<block,64,affine,level8<SUBNET>> >;

using anet_type = loss_metric<
                      fc_no_bias<128,
                        avg_pool_everything<
                          level9<
                            max_pool<3,3,2,2,
                              relu<affine<con<64,7,7,2,2,
                                input_rgb_image_sized<150>
                              >>>
                            >
                          >
                        >
                      >
                    >;

// ---------------------------------------------
// 2) 内部 Impl 结构体，隐藏所有实现细节
// ---------------------------------------------
struct FaceRecognizer::Impl {
    std::string configPath;
    double threshold = 0.6;
    shape_predictor sp;
    anet_type net;
    std::vector<std::string> names;
    std::vector<matrix<float,0,1>> embeddings;

    // 检测器可以复用
    frontal_face_detector detector = get_frontal_face_detector();
};

} // anonymous namespace

// ---------------------------------------------
// 3) FaceRecognizer 接口实现
// ---------------------------------------------
FaceRecognizer::FaceRecognizer(const std::string& cfg)
 : impl_(new Impl())
{
    impl_->configPath = cfg;
}

FaceRecognizer::~FaceRecognizer() = default;

bool FaceRecognizer::init() {
    ConfigParser cfg(impl_->configPath);
    if (!cfg.load()) {
        std::cerr << "Failed to load config: " << impl_->configPath << std::endl;
        return false;
    }

    // 读取阈值和模型路径
    impl_->threshold =
        std::stod(cfg.get("face_threshold", "0.6"));
    std::string spath = cfg.get("shape_model");
    std::string dpath = cfg.get("dnn_model");
    std::string libCsv = cfg.get("face_lib_csv");

    // 加载模型
    try {
        deserialize(spath) >> impl_->sp;
        deserialize(dpath) >> impl_->net;
    } catch (std::exception& e) {
        std::cerr << "Model load error: " << e.what() << std::endl;
        return false;
    }

    // 加载人脸库 CSV：每行 name v0 v1 ... v127
    if (!libCsv.empty()) {
        std::ifstream fin(libCsv);
        if (!fin) {
            std::cerr << "Cannot open face library: " << libCsv << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(fin, line)) {
            std::istringstream iss(line);
            std::string name;
            iss >> name;
            matrix<float,0,1> emb;
            emb.set_size(128);
            for (int i = 0; i < 128; ++i) iss >> emb(i);
            impl_->names.push_back(name);
            impl_->embeddings.push_back(emb);
        }
    }

    return true;
}

void FaceRecognizer::printLibrary() const {
    std::cout << "人脸库信息:" << std::endl;
    for (size_t i = 0; i < impl_->names.size(); ++i) {
        std::cout << "  " << impl_->names[i]
                  << " (dim=" << impl_->embeddings[i].nr() << ")" 
                  << std::endl;
    }
}

std::string FaceRecognizer::recognize(const matrix<rgb_pixel>& img) {
    // 1) 检测第一张人脸
    auto dets = impl_->detector(img);
    if (dets.empty()) return "";

    // 2) 关键点与对齐
    auto shape = impl_->sp(img, dets[0]);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img,
        get_face_chip_details(shape,150,0.25),
        face_chip);

    // 3) 生成特征向量
    auto face_desc = impl_->net(face_chip);

    // 4) 在库中找最小距离
    double best = std::numeric_limits<double>::max();
    int bestIdx = -1;
    for (size_t i = 0; i < impl_->embeddings.size(); ++i) {
        double dist = length(impl_->embeddings[i] - face_desc);
        if (dist < best) {
            best = dist;
            bestIdx = int(i);
        }
    }
    if (bestIdx >= 0 && best < impl_->threshold)
        return impl_->names[bestIdx];
    return "";
}

