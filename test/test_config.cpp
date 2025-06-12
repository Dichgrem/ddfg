#include "ConfigParser.h"
#include <iostream>
#include <filesystem>
#include <string>

int main() {
    ConfigParser config;
    // 请确保路径正确
    if (!config.load("../config/config.json")) {
        return -1;
    }

    // 1. 打印所有配置项
    std::cout << "--- Printing all configurations ---" << std::endl;
    config.printAll();
    std::cout << std::endl;

    // 2. 检查路径有效性
    std::cout << "--- Verifying paths from config ---" << std::endl;
    auto path_keys = config.getPathKeys();
    
    // 如果使用摄像头，则跳过video_path的检查
    bool use_camera = config.get<bool>("use_camera", true);

    for (const auto& key : path_keys) {
        if (key == "video_path" && use_camera) {
            std::cout << "Skipping 'video_path' check as camera is in use." << std::endl;
            continue;
        }

        std::string path_str = config.get<std::string>(key, "");
        if (path_str.empty()) {
            std::cerr << "Path for key '" << key << "' is empty. Skipping check." << std::endl;
            continue;
        }

        if (std::filesystem::exists(path_str)) {
            std::cout << "Path check PASSED for '" << key << "': " << path_str << std::endl;
        } else {
            std::cerr << "Path check FAILED for '" << key << "': " << path_str << " does not exist." << std::endl;
        }
    }
     std::cout << "--- Verification finished ---" << std::endl;

    return 0;
}
