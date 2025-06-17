#pragma once
#include <opencv2/opencv.hpp>
#include "Matrix.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

class ImageUtils {
public:
    static cv::Mat load_image(const std::string& path, bool grayscale = true) {
        int flags = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        cv::Mat image = cv::imread(path, flags);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + path);
        }
        return image;
    }
    
    static Vector<float> preprocess_face(const cv::Mat& face) {
        // Convert to float and normalize
        cv::Mat processed;
        face.convertTo(processed, CV_32F, 1.0/255.0);
        
        // Flatten to vector
        Vector<float> result(1, face.rows * face.cols);
        memcpy(result.get_data().data(), processed.data, 
               face.rows * face.cols * sizeof(float));
        return result;
    }
    
    static cv::Rect detect_face(const cv::Mat& frame) {
        // Simple face detection (center crop) for deployment
        int w = frame.cols * 0.7;
        int h = frame.rows * 0.7;
        int x = (frame.cols - w) / 2;
        int y = (frame.rows - h) / 2;
        return cv::Rect(x, y, w, h);
    }
    
    static void draw_result(cv::Mat& frame, const cv::Rect& face, 
                           const std::string& emotion, float confidence) {
        // Draw face rectangle
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        
        // Draw emotion label
        std::string label = emotion + ": " + std::to_string(int(confidence * 100)) + "%";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        
        cv::rectangle(frame, 
                     cv::Point(face.x, face.y - text_size.height - 10),
                     cv::Point(face.x + text_size.width, face.y),
                     cv::Scalar(0, 0, 0), cv::FILLED);
        
        cv::putText(frame, label, cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    
    static Matrix<float> load_dataset(const std::string& dir_path) {
        std::vector<std::string> file_paths;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.path().extension() == ".pgm" || 
                entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".png") {
                file_paths.push_back(entry.path().string());
            }
        }
        
        if (file_paths.empty()) {
            throw std::runtime_error("No images found in: " + dir_path);
        }
        
        // Read first image to get dimensions
        cv::Mat sample = load_image(file_paths[0]);
        size_t num_features = sample.rows * sample.cols;
        
        Matrix<float> dataset(file_paths.size(), num_features);
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < file_paths.size(); ++i) {
            cv::Mat img = load_image(file_paths[i]);
            Vector<float> features = preprocess_face(img);
            dataset.set_row(i, features.get_data());
        }
        
        return dataset;
    }
    
    static cv::Mat crop_face(const cv::Mat& frame, const cv::Rect& face_roi) {
        return frame(face_roi).clone();
    }
    
    static cv::Mat resize_face(const cv::Mat& face, int width = 64, int height = 64) {
        cv::Mat resized;
        cv::resize(face, resized, cv::Size(width, height));
        return resized;
    }
};