#include "FaceRecognizer.h"
#include <iostream>

// Impl的完整定义
struct FaceRecognizer::Impl {
    std::string configPath;
    // 其它成员，比如网络模型，参数等
    std::vector<std::string> names;
};

FaceRecognizer::FaceRecognizer(const std::string& configPath)
    : impl_(new Impl())  // 在cpp文件里创建Impl实例
{
    impl_->configPath = configPath;
}

FaceRecognizer::~FaceRecognizer() {
    delete impl_;  // 释放impl
}

bool FaceRecognizer::init() {
    // 初始化代码，比如加载模型等
    impl_->names.push_back("example");
    return true;
}

std::string FaceRecognizer::recognize(const dlib::matrix<dlib::rgb_pixel>& img) {
    // 使用impl_
    if (!impl_->names.empty())
        return impl_->names[0];
    return "";
}

void FaceRecognizer::printLibrary() const {
    for (const auto& name : impl_->names) {
        std::cout << name << std::endl;
    }
}

