#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "Matrix.hpp"
#include "PCA.hpp"
#include "GMM.hpp"
#include "FFNN.hpp"
#include "ImageUtils.hpp"
#include "Utils.hpp"   // for parseCLI, readBinary, Timer

int main(int argc, char** argv) {
    // Parse CLI
    auto cli = Utils::parseCLI(argc, argv);

    std::cout << "[Main] Mode: " << cli.mode
              << "  Data dir: " << cli.data_dir;
    if (!cli.test_image.empty())
        std::cout << "  Test image: " << cli.test_image;
    std::cout << "  Camera: " << cli.camera_index << "\n";

    // 1. Load PCA parameters
    std::string mean_file = cli.data_dir + "/mean_face.bin";
    std::string eig_file  = cli.data_dir + "/eigenvectors.bin";

    std::vector<double> mean, eig_flat;
    if (!Utils::readBinary(mean_file, mean)) {
        std::cerr << "Error: failed to load " << mean_file << "\n";
        return -1;
    }
    if (!Utils::readBinary(eig_file, eig_flat)) {
        std::cerr << "Error: failed to load " << eig_file << "\n";
        return -1;
    }

    const int D = static_cast<int>(mean.size());
    const int k = static_cast<int>(eig_flat.size() / D);
    std::vector<std::vector<double>> eigenvectors(k, std::vector<double>(D));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < D; ++j)
            eigenvectors[i][j] = eig_flat[i * D + j];

    // 2. Load GMM
    GMM::GMMModel<double> gmm(/*num_components=*/4, /*dim=*/k);
    try {
        gmm.loadFromFile(cli.data_dir + "/gmm_params.bin");
    } catch (const std::exception& ex) {
        std::cerr << "Error loading GMM params: " << ex.what() << "\n";
        return -1;
    }

    // 3. Load FFNN
    FFNN::Network<double> net;
    const int H = 32, C = 4;
    net.init(k, H, C);
    if (!net.loadParameters(
            cli.data_dir + "/ffnn_w1.bin",
            cli.data_dir + "/ffnn_b1.bin",
            cli.data_dir + "/ffnn_w2.bin",
            cli.data_dir + "/ffnn_b2.bin"))
    {
        std::cerr << "Error: failed to load FFNN parameters\n";
        return -1;
    }

    // If test-image mode, run once and save output
    if (!cli.test_image.empty()) {
        std::vector<double> flat;
        int w, h;
        if (!ImageUtils::loadPGMAsVector<double>(cli.test_image, flat, w, h)) {
            std::cerr << "Error: cannot load test image: " << cli.test_image << "\n";
            return -1;
        }
        // preprocess
        for (int i = 0; i < D && i < (int)flat.size(); ++i)
            flat[i] -= mean[i];
        auto feat = PCA::project<double>(flat, eigenvectors);
        auto post = gmm.infer(feat);
        auto probs = net.forward(feat);
        int label_idx = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        static const char* labels[] = {"Angry","Happy","Sad","Neutral"};
        std::cout << "Predicted emotion: " << labels[label_idx] << "\n";

        // write out the original as JPEG to indicate success
        cv::Mat img(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.at<uchar>(y, x) = static_cast<uchar>(flat[y*w + x] + mean[y*w + x]);
        std::string outpath = cli.data_dir + "/output.jpg";
        cv::imwrite(outpath, img);
        std::cout << "Wrote output image to " << outpath << "\n";
        return 0;
    }

    // Otherwise, real-time inference
    if (cli.mode != "inference") {
        std::cerr << "Error: only 'inference' and 'test-image' modes are supported.\n";
        return -1;
    }

    cv::VideoCapture cap(cli.camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open camera " << cli.camera_index << "\n";
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto faces = ImageUtils::detectFacesHaar(frame,
                     cli.data_dir + "/haarcascade_frontalface_default.xml");
        if (!faces.empty()) {
            cv::Rect roi = faces[0];
            std::vector<double> flat;
            if (ImageUtils::extractFaceVector<double>(frame, roi, flat, 64, 64)) {
                for (int i = 0; i < D; ++i) flat[i] -= mean[i];
                auto feat = PCA::project<double>(flat, eigenvectors);
                auto probs = net.forward(feat);
                int idx = std::distance(probs.begin(),
                                        std::max_element(probs.begin(), probs.end()));
                static const char* labels[] = {"Angry","Happy","Sad","Neutral"};
                ImageUtils::drawLabel(frame, roi, labels[idx]);
            }
        }
        cv::imshow("Emotion Recognition", frame);
        if (cv::waitKey(1) == 27) break;  // ESC
    }

    return 0;
}
