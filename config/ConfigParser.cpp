#include "ConfigParser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>

ConfigParser::ConfigParser(const std::string& filepath)
    : _path(filepath) {}

bool ConfigParser::load() {
    try {
        YAML::Node root = YAML::LoadFile(_path);
        for (auto it : root) {
            _items[it.first.as<std::string>()] = it.second.as<std::string>();
        }
    } catch (std::exception &e) {
        std::cerr << "Load config error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

std::string ConfigParser::get(const std::string& key) const {
    auto it = _items.find(key);
    return (it != _items.end()) ? it->second : std::string{};
}

void ConfigParser::printAll() const {
    for (auto& [k, v] : _items) {
        std::cout << k << " = " << v << std::endl;
    }
}

