#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <vector>
#include "nlohmann/json.hpp" // 引入json解析库

using json = nlohmann::json;

class ConfigParser {
public:
    // 加载并解析配置文件
    bool load(const std::string& config_path);

    // 获取指定配置项的值 (模板函数，支持string, bool, double, int等)
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const;

    // 打印所有配置项
    void printAll() const;

    // 获取所有包含路径的键
    std::vector<std::string> getPathKeys() const;


private:
    json config_data_;
};

#endif // CONFIG_PARSER_H
