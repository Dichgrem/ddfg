#include "../config/ConfigParser.h"
#include <filesystem>
#include <iostream>

int main() {
    ConfigParser cfg("../config.yaml");
    if (!cfg.load()) {
        std::cerr << "Failed to load config\n"; return -1;
    }
    cfg.printAll();

    for (auto& key : {"video_path", "shape_model", "dnn_model", "face_lib_csv"}) {
        std::string v = cfg.get(key);
        std::cout << "检查 " << key << ": " << v << " ... "
                  << (std::filesystem::exists(v) ? "存在" : "不存在") << std::endl;
    }
    return 0;
}

