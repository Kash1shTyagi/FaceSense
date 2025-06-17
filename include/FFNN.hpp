#pragma once
#include "Matrix.hpp"
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cmath>

class FFNN {
public:
    FFNN() = default;
    
    void load_weights(const std::string& w1_path, const std::string& b1_path,
                      const std::string& w2_path, const std::string& b2_path) {
        w1_ = Matrix<float>::load(w1_path);
        b1_ = Vector<float>::load(b1_path);
        w2_ = Matrix<float>::load(w2_path);
        b2_ = Vector<float>::load(b2_path);
    }
    
    void load_weights(std::istream& w1_stream, std::istream& b1_stream,
                      std::istream& w2_stream, std::istream& b2_stream) {
        w1_ = Matrix<float>::load(w1_stream);
        b1_ = Vector<float>::load(b1_stream);
        w2_ = Matrix<float>::load(w2_stream);
        b2_ = Vector<float>::load(b2_stream);
    }
    
    std::vector<float> predict(const Vector<float>& input) const {
        // Layer 1: ReLU activation
        Vector<float> hidden = w1_ * input.transpose();
        hidden = hidden + b1_;
        relu(hidden);
        
        // Layer 2: Linear + Softmax
        Vector<float> output = w2_ * hidden;
        output = output + b2_;
        softmax(output);
        
        // Convert to std::vector
        std::vector<float> result(output.num_cols());
        for (size_t i = 0; i < output.num_cols(); ++i) {
            result[i] = output(0, i);
        }
        return result;
    }
    
    // For testing
    const Matrix<float>& w1() const { return w1_; }
    const Vector<float>& b1() const { return b1_; }
    const Matrix<float>& w2() const { return w2_; }
    const Vector<float>& b2() const { return b2_; }
    
private:
    Matrix<float> w1_, w2_;
    Vector<float> b1_, b2_;
    
    static void relu(Vector<float>& v) {
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (size_t i = 0; i < v.num_cols(); ++i) {
            v(0, i) = std::max(0.0f, v(0, i));
        }
    }
    
    static void softmax(Vector<float>& v) {
        // Find max for numerical stability
        float max_val = *std::max_element(v.get_data().begin(), v.get_data().end());
        
        // Exponentiate and sum
        float sum = 0.0f;
        #ifdef _OPENMP
        #pragma omp simd reduction(+:sum)
        #endif
        for (size_t i = 0; i < v.num_cols(); ++i) {
            v(0, i) = std::exp(v(0, i) - max_val);
            sum += v(0, i);
        }
        
        // Normalize
        if (sum > 0) {
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (size_t i = 0; i < v.num_cols(); ++i) {
                v(0, i) /= sum;
            }
        }
    }
};