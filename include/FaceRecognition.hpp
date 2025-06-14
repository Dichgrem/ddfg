#ifndef FACE_RECOGNITION_HPP
#define FACE_RECOGNITION_HPP

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <string>
#include <unordered_map>

// 前向声明
class ConfigParser;

// 使用 dlib 的标准人脸识别网络定义
// 这是 dlib 官方推荐的人脸识别网络结构
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

// 定义人脸识别网络
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

class FaceRecognition
{
public:
    explicit FaceRecognition(const ConfigParser& config);
    
    // 识别人脸
    std::string recognize(const dlib::matrix<dlib::rgb_pixel>& face_chip);
    
    // 获取形状预测器
    dlib::shape_predictor getShapePredictor() const;
    
    // 打印人脸库信息
    void printFaceLibInfo() const;

private:
    // 加载模型
    void loadModels(const ConfigParser& config);
    
    // 从CSV加载人脸库
    void loadLibraryFromCSV(const std::string& csv_path);
    
    // 从目录构建人脸库
    void buildFaceLibrary(const std::string& dir_path);

private:
    anet_type net_;                           // 人脸识别网络
    dlib::shape_predictor sp_;                // 形状预测器
    double face_match_threshold_;             // 人脸匹配阈值
    
    // 人脸库：姓名 -> 特征向量
    std::unordered_map<std::string, dlib::matrix<float,0,1>> face_library_;
};

#endif // FACE_RECOGNITION_HPP

