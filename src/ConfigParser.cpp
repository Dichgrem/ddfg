#include "ConfigParser.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm> // 需要包含 <algorithm> 来使用 std::replace

bool ConfigParser::load(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        std::cerr << "Error: Failed to open config file: " << config_path << std::endl;
        return false;
    }
    try {
        config_data_ = json::parse(f);
    } catch (json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

template<typename T>
T ConfigParser::get(const std::string& key, const T& default_value) const {
    try {
        // --- START OF FIX ---
        // 1. 先创建一个可修改的字符串
        std::string path_key = key;

        // 2. 在字符串上执行替换操作
        std::replace(path_key.begin(), path_key.end(), '.', '/');

        // 3. 用修正后的字符串创建 json_pointer
        json::json_pointer ptr("/" + path_key);
        // --- END OF FIX ---

        return config_data_.at(ptr).get<T>();
    } catch (json::out_of_range&) {
        // Key not found
        return default_value;
    } catch (json::type_error&) {
        // Key found but type is wrong
        return default_value;
    }
}

// 显式实例化模板
template std::string ConfigParser::get<std::string>(const std::string&, const std::string&) const;
template bool ConfigParser::get<bool>(const std::string&, const bool&) const;
template double ConfigParser::get<double>(const std::string&, const double&) const;
template int ConfigParser::get<int>(const std::string&, const int&) const;

void ConfigParser::printAll() const {
    if (config_data_.empty()) {
        std::cout << "Config is empty." << std::endl;
        return;
    }
    std::cout << "All configurations:" << std::endl;
    std::cout << config_data_.dump(4) << std::endl; // 4 = indent size
}

std::vector<std::string> ConfigParser::getPathKeys() const {
    // 定义哪些key是路径，需要检查
    // 注意：嵌套的键在这里也需要用点号表示
    return {"video_path", "models.shape_predictor", "models.face_recognition", "face_library_path"};
}
