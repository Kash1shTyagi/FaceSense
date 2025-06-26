#pragma once
#include <string>
#include <chrono>
#include <vector>

namespace Utils {

    // A simple struct for command-line options
    struct CLI {
        std::string mode;         // e.g. "inference", "train-pca", ...
        std::string data_dir;     // path to data directory
        int         camera_index; // default 0
        std::string test_image;  
    };

    /**
     * @brief Parse argc/argv into a CLI struct.
     * Recognized flags:
     *   -m | --mode <string>
     *   -d | --data-dir <string>
     *   -c | --camera <int>
     */
    CLI parseCLI(int argc, char** argv);

    // Binary I/O for POD vectors (float or double)
    template<typename T>
    bool readBinary(const std::string& path, std::vector<T>& out);

    template<typename T>
    bool writeBinary(const std::string& path, const std::vector<T>& data);

    // Timer utility (prints elapsed ms when destructed)
    class Timer {
    public:
        Timer(const std::string& name);
        ~Timer();
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

}  // namespace Utils
