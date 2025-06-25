#pragma once
#include <vector>
#include <string>

/**
 * @brief Small Feedforward Neural Network inference in C++.
 * Architecture: input size k → hidden size H (e.g., 32) with ReLU → output size C (num classes) with Softmax.
 * Loads weights/biases exported from Python.
 */
namespace FFNN {

    /**
     * @brief Activation: ReLU
     */
    template<typename T>
    void relu(std::vector<T>& x) {
        for (auto& xi : x)
            xi = xi > T{0} ? xi : T{0};
    }

    /**
     * @brief Softmax in-place. Subtract max for numerical stability.
     * @param x: logits vector, length C; replaced by probabilities summing to 1.
     */
    template<typename T>
    void softmax(std::vector<T>& x) {
        T maxv = x.empty() ? T{} : x[0];
        for (auto& xi : x)
            if (xi > maxv) maxv = xi;
        T sum = T{};
        for (auto& xi : x) {
            xi = std::exp(xi - maxv);
            sum += xi;
        }
        if (sum <= T{}) return;
        for (auto& xi : x)
            xi /= sum;
    }

    /**
     * @brief Load binary file into vector<T>. Implement in Utils.cpp: open file in binary mode, read floats/doubles.
     * E.g., for float: reinterpret_cast<float*> buffer; for double: similar.
     * @param filename: path to .bin file
     * @param out: vector<T> to fill
     */
    template<typename T>
    bool loadBinary(const std::string& filename, std::vector<T>& out);

    /**
     * @brief FFNN inference engine.
     */
    template<typename T>
    class Network {
    public:
        Network() : k(0), H(0), C(0) {}
        /**
         * @brief Initialize sizes and allocate weight/bias containers.
         */
        void init(size_t input_dim, size_t hidden_dim, size_t output_dim) {
            k = input_dim;
            H = hidden_dim;
            C = output_dim;
            W1.assign(H * k, T{});  // row-major: H rows, k cols
            b1.assign(H, T{});
            W2.assign(C * H, T{});
            b2.assign(C, T{});
        }
        /**
         * @brief Load parameters from binary files.
         * Expect row-major float/double arrays in agreed format.
         */
        bool loadParameters(const std::string& w1_file,
                            const std::string& b1_file,
                            const std::string& w2_file,
                            const std::string& b2_file) {
            if (!loadBinary(w1_file, W1) ||
                !loadBinary(b1_file, b1) ||
                !loadBinary(w2_file, W2) ||
                !loadBinary(b2_file, b2)) {
                return false;
            }
            // Optionally check sizes correct: W1.size() == H*k, etc.
            return true;
        }

        /**
         * @brief Forward pass: input x (length k) → output probabilities (length C).
         */
        std::vector<T> forward(const std::vector<T>& x) const {
            if (x.size() != k) throw std::runtime_error("Input dimension mismatch in FFNN::forward");
            // Hidden: h = ReLU(W1 * x + b1)
            std::vector<T> h(H, T{});
            for (size_t i = 0; i < H; ++i) {
                T sum = T{};
                for (size_t j = 0; j < k; ++j) {
                    sum += W1[i * k + j] * x[j];
                }
                sum += b1[i];
                h[i] = sum > T{0} ? sum : T{0};
            }
            // Output logits: z = W2 * h + b2
            std::vector<T> z(C, T{});
            for (size_t i = 0; i < C; ++i) {
                T sum = T{};
                for (size_t j = 0; j < H; ++j) {
                    sum += W2[i * H + j] * h[j];
                }
                sum += b2[i];
                z[i] = sum;
            }
            // Softmax
            softmax(z);
            return z;
        }

    private:
        size_t k, H, C;
        std::vector<T> W1;  // size H*k
        std::vector<T> b1;  // size H
        std::vector<T> W2;  // size C*H
        std::vector<T> b2;  // size C
    };

} // namespace FFNN
