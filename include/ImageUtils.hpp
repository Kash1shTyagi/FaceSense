#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Utilities for image I/O, conversion between OpenCV Mat and flattened vectors, face detection.
 */
namespace ImageUtils {

    /**
     * @brief Load a PGM (grayscale) image from file into a flattened vector.
     * @param filepath: path to .pgm file.
     * @param out: vector to fill with pixel values normalized or raw (e.g., 0-255 as double/float).
     * @param width, height: expected dimensions; if zero, read from image.
     * @return true if success.
     */
    template<typename T>
    bool loadPGMAsVector(const std::string& filepath, std::vector<T>& out, int& width, int& height) {
        cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            return false;
        }
        width = img.cols;
        height = img.rows;
        out.resize(width * height);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                T val = static_cast<T>(img.at<uchar>(r, c));
                out[r * width + c] = val;
            }
        }
        return true;
    }

    /**
     * @brief Given OpenCV Mat frame and a bounding box, extract grayscale, resized face vector.
     * @param frame: BGR or grayscale Mat.
     * @param face_rect: bounding box for face in frame.
     * @param out: flattened vector<T> length resize_w * resize_h.
     * @param resize_w, resize_h: e.g., 64, 64.
     * @return true if ROI valid and extraction succeeded.
     */
    template<typename T>
    bool extractFaceVector(const cv::Mat& frame, const cv::Rect& face_rect,
                           std::vector<T>& out, int resize_w = 64, int resize_h = 64) {
        if (frame.empty() || face_rect.width <= 0 || face_rect.height <= 0)
            return false;
        // Crop ROI with boundary checks
        cv::Rect roi = face_rect & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width <= 0 || roi.height <= 0)
            return false;
        cv::Mat face = frame(roi);
        cv::Mat gray;
        if (face.channels() == 3)
            cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        else
            gray = face;
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(resize_w, resize_h));
        out.resize(resize_w * resize_h);
        for (int r = 0; r < resize_h; ++r) {
            for (int c = 0; c < resize_w; ++c) {
                T val = static_cast<T>(resized.at<uchar>(r, c));
                out[r * resize_w + c] = val;
            }
        }
        return true;
    }

    /**
     * @brief Simple face detector using Haar cascades.
     * Load cascade once and detect faces in frame.
     * @param cascade_path: path to haarcascade xml.
     * @param frame: input BGR frame.
     * @return vector of cv::Rect for detected faces.
     */
    inline std::vector<cv::Rect> detectFacesHaar(const cv::Mat& frame, const std::string& cascade_path) {
        std::vector<cv::Rect> faces;
        static cv::CascadeClassifier face_cascade;
        static bool loaded = false;
        if (!loaded) {
            if (!face_cascade.load(cascade_path)) {
                throw std::runtime_error("Failed to load Haar cascade from " + cascade_path);
            }
            loaded = true;
        }
        cv::Mat gray;
        if (frame.channels() == 3)
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else
            gray = frame;
        cv::equalizeHist(gray, gray);
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        return faces;
    }

    /**
     * @brief Draw bounding box and label on frame.
     */
    inline void drawLabel(cv::Mat& frame, const cv::Rect& rect, const std::string& label) {
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.7;
        int thickness = 2;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg(rect.x, rect.y - 5);
        if (textOrg.y < textSize.height) textOrg.y = rect.y + textSize.height + 5;
        cv::putText(frame, label, textOrg, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
    }

    /**
     * @brief Read/write binary vectors or matrices.
     * Implement in Utils.cpp if desired.
     */
} // namespace ImageUtils
