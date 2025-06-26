#include "Utils.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>

// Explicit instantiations of the templates
template bool Utils::readBinary<double>(const std::string&, std::vector<double>&);
template bool Utils::writeBinary<double>(const std::string&, const std::vector<double>&);

namespace Utils {

    bool readBinary(const std::string& path, std::vector<double>& out) = delete;
    bool writeBinary(const std::string& path, const std::vector<double>& data) = delete;

    template<typename T>
    bool readBinary(const std::string& path, std::vector<T>& out) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "[Utils] Failed to open for reading: " << path << "\n";
            return false;
        }
        auto fileSize = std::filesystem::file_size(path);
        size_t count = fileSize / sizeof(T);
        out.resize(count);
        in.read(reinterpret_cast<char*>(out.data()), count * sizeof(T));
        if (!in) {
            std::cerr << "[Utils] Read error in: " << path << "\n";
            return false;
        }
        return true;
    }

    template<typename T>
    bool writeBinary(const std::string& path, const std::vector<T>& data) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "[Utils] Failed to open for writing: " << path << "\n";
            return false;
        }
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        if (!out) {
            std::cerr << "[Utils] Write error in: " << path << "\n";
            return false;
        }
        return true;
    }

    Timer::Timer(const std::string& name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    Timer::~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "[Timer] " << name_ << " took " << ms << " ms\n";
    }

    CLI parseCLI(int argc, char** argv) {
        CLI cli;
        cli.mode         = "inference";
        cli.data_dir     = "data";
        cli.camera_index = 0;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
                cli.mode = argv[++i];
            }
            else if ((arg == "-d" || arg == "--data-dir") && i + 1 < argc) {
                cli.data_dir = argv[++i];
            }
            else if ((arg == "-c" || arg == "--camera") && i + 1 < argc) {
                cli.camera_index = std::stoi(argv[++i]);
            }
            else if (arg == "--test-image" && i + 1 < argc) {
                 cli.test_image = argv[++i];
            }
            else {
                std::cerr << "[Utils] Unknown or incomplete arg: " << arg << "\n";
            }
        }
        return cli;
    }

}  // namespace Utils
