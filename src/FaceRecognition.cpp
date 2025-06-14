#include "FaceRecognition.hpp"
#include "ConfigParser.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace dr = dlib;

FaceRecognition::FaceRecognition(const ConfigParser& config)
{
    std::cout << "Initializing FaceRecognition module..." << std::endl;
    loadModels(config);

    face_match_threshold_ = config.get<double>("face_match_threshold", 0.6);

    bool use_csv = config.get<bool>("face_lib.use_csv", false);
    if (use_csv)
    {
        const auto csv_path = config.get<std::string>("face_lib.csv_path", "");
        loadLibraryFromCSV(csv_path);
    }
    else
    {
        const auto dir_path = config.get<std::string>("face_lib.dir_path", "");
        buildFaceLibrary(dir_path);
    }
}

void FaceRecognition::loadModels(const ConfigParser& config)
{
    const auto sp_path  = config.get<std::string>("models.shape_predictor", "");
    const auto net_path = config.get<std::string>("models.face_recognition", "");
    if (sp_path.empty() || net_path.empty())
        throw std::runtime_error("Model paths missing in config.");

    dr::deserialize(sp_path) >> sp_;
    std::cout << "Shape predictor loaded from: " << sp_path << std::endl;

    dr::deserialize(net_path) >> net_;
    std::cout << "Face recognition model loaded from: " << net_path << std::endl;
}

void FaceRecognition::loadLibraryFromCSV(const std::string& csv_path)
{
    if (csv_path.empty() || !std::filesystem::exists(csv_path))
    {
        std::cerr << "CSV path not found: " << csv_path << std::endl;
        return;
    }

    std::ifstream fin(csv_path);
    std::string line;
    size_t count = 0;

    while (std::getline(fin, line))
    {
        std::istringstream iss(line);
        std::string name;
        if (!std::getline(iss, name, ',')) continue;

        std::vector<float> values;
        float v;
        while (iss >> v) values.push_back(v);

        if (values.empty()) continue;
        dr::matrix<float,0,1> desc(values.size());
        for (size_t i = 0; i < values.size(); ++i)
            desc(i) = values[i];

        face_library_[name] = desc;
        ++count;
    }
    std::cout << "Loaded " << count << " entries from CSV library." << std::endl;
}

void FaceRecognition::buildFaceLibrary(const std::string& dir_path)
{
    if (dir_path.empty() || !std::filesystem::exists(dir_path))
    {
        std::cerr << "Directory path not found: " << dir_path << std::endl;
        return;
    }

    auto detector = dr::get_frontal_face_detector();
    size_t count = 0;

    for (const auto& person_dir : std::filesystem::directory_iterator(dir_path))
    {
        if (!person_dir.is_directory()) continue;
        const auto name = person_dir.path().filename().string();

        for (const auto& img_file : std::filesystem::directory_iterator(person_dir))
        {
            dr::matrix<dr::rgb_pixel> img;
            try
            {
                dr::load_image(img, img_file.path().string());
            }
            catch (const std::exception& e)
            {
                std::cerr << "Failed loading image " << img_file.path()
                          << ": " << e.what() << std::endl;
                continue;
            }

            auto faces = detector(img);
            if (faces.size() != 1)
            {
                std::cerr << "Skipping " << img_file.path()
                          << ": found " << faces.size() << " faces." << std::endl;
                continue;
            }

            auto shape = sp_(img, faces[0]);
            dr::matrix<dr::rgb_pixel> chip;
            dr::extract_image_chip(img,
                dr::get_face_chip_details(shape, 150, 0.25),
                chip);

            auto desc = net_(chip);
            face_library_[name] = desc;
            ++count;
            break;  // 每个子目录只用第一张有效图片
        }
    }
    std::cout << "Built face library from directory: " << count << " entries." << std::endl;
}

std::string FaceRecognition::recognize(const dr::matrix<dr::rgb_pixel>& face_chip)
{
    if (face_library_.empty())
        return "Stranger";

    auto descriptor = net_(face_chip);
    double min_dist = std::numeric_limits<double>::infinity();
    std::string best = "Stranger";

    for (const auto& [name, lib_desc] : face_library_)
    {
        double d = dlib::length(lib_desc - descriptor);
        if (d < min_dist)
        {
            min_dist = d;
            best = name;
        }
    }

    return (min_dist <= face_match_threshold_) ? best : "Stranger";
}

void FaceRecognition::printFaceLibInfo() const
{
    std::cout << "----- Face Library Info -----\n";
    std::cout << "Total entries : " << face_library_.size() << "\n";
    std::cout << "Threshold     : " << face_match_threshold_ << "\n";
    for (const auto& [name, desc] : face_library_)
    {
        std::cout << " - " << name << " (dim=" << desc.size() << ")\n";
    }
    std::cout << "-----------------------------\n";
}

dr::shape_predictor FaceRecognition::getShapePredictor() const
{
    return sp_;
}

