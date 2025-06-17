#pragma once
#include "Matrix.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>

class PCA {
public:
    PCA() = default;
    
    void train(const Matrix<float>& data, size_t components, size_t max_iter = 1000, float tol = 1e-6f) {
        // Compute mean
        mean_ = compute_mean(data);
        
        // Center data
        Matrix<float> centered = center_data(data);
        
        // Compute covariance
        Matrix<float> cov = compute_covariance(centered);
        
        // Compute eigenvectors using power iteration
        eigenvectors_ = Matrix<float>(components, data.num_cols());
        eigenvalues_.resize(components);
        
        Matrix<float> cov_copy = cov;  // Working copy
        
        for (size_t i = 0; i < components; ++i) {
            auto [eigenvalue, eigenvector] = power_iteration(cov_copy, max_iter, tol);
            eigenvalues_[i] = eigenvalue;
            set_eigenvector_row(i, eigenvector);
            deflate(cov_copy, eigenvalue, eigenvector);
        }
    }
    
    Matrix<float> project(const Matrix<float>& data) const {
        Matrix<float> centered = center_data(data);
        return centered * eigenvectors_.transpose();
    }
    
    // Serialization
    void save(const std::string& prefix) const {
        mean_.save(prefix + "_mean.bin");
        eigenvectors_.save(prefix + "_eigenvectors.bin");
        
        std::ofstream eval_file(prefix + "_eigenvalues.bin", std::ios::binary);
        eval_file.write(reinterpret_cast<const char*>(eigenvalues_.data()), 
                        eigenvalues_.size() * sizeof(float));
    }
    
    void load(const std::string& prefix) {
        mean_.load(prefix + "_mean.bin");
        eigenvectors_.load(prefix + "_eigenvectors.bin");
        
        std::ifstream eval_file(prefix + "_eigenvalues.bin", std::ios::binary);
        if (!eval_file) throw std::runtime_error("Cannot open eigenvalues file");
        
        size_t size = eigenvectors_.num_rows();
        eigenvalues_.resize(size);
        eval_file.read(reinterpret_cast<char*>(eigenvalues_.data()), 
                       size * sizeof(float));
    }
    
    // For testing
    const Matrix<float>& eigenvectors() const { return eigenvectors_; }
    const std::vector<float>& eigenvalues() const { return eigenvalues_; }
    const Matrix<float>& mean() const { return mean_; }
    
private:
    Matrix<float> mean_;
    Matrix<float> eigenvectors_;
    std::vector<float> eigenvalues_;
    
    void set_eigenvector_row(size_t i, const Matrix<float>& row) {
        if (row.num_rows() != 1) {
            throw std::invalid_argument("Expected row vector");
        }
        for (size_t j = 0; j < eigenvectors_.num_cols(); j++) {
            eigenvectors_(i, j) = row(0, j);
        }
    }
    
    Matrix<float> compute_mean(const Matrix<float>& data) const {
        Matrix<float> mean(1, data.num_cols());
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t j = 0; j < data.num_cols(); ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < data.num_rows(); ++i) {
                sum += data(i, j);
            }
            mean(0, j) = sum / data.num_rows();
        }
        return mean;
    }
    
    Matrix<float> center_data(const Matrix<float>& data) const {
        Matrix<float> centered(data.num_rows(), data.num_cols());
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < data.num_rows(); ++i) {
            for (size_t j = 0; j < data.num_cols(); ++j) {
                centered(i, j) = data(i, j) - mean_(0, j);
            }
        }
        return centered;
    }
    
    Matrix<float> compute_covariance(const Matrix<float>& centered) const {
        return centered.transpose() * centered * (1.0f / (centered.num_rows() - 1));
    }
    
    std::pair<float, Matrix<float>> power_iteration(const Matrix<float>& A, 
                                                   size_t max_iter, 
                                                   float tol) const {
        size_t n = A.num_cols();
        Matrix<float> v(1, n);  // Row vector
        
        // Initialize with random vector
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            v(0, i) = dist(gen);
        }
        
        // Normalize
        float norm = v.norm();
        for (size_t i = 0; i < n; ++i) {
            v(0, i) /= norm;
        }
        
        Matrix<float> v_prev;
        float eigenvalue = 0.0f;
        
        for (size_t iter = 0; iter < max_iter; ++iter) {
            v_prev = v;
            
            // Matrix-vector multiplication: v = A * v^T
            Matrix<float> temp(n, 1);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    temp(i, 0) += A(i, j) * v(0, j);
                }
            }
            
            // Convert back to row vector
            v = Matrix<float>(1, n);
            for (size_t i = 0; i < n; ++i) {
                v(0, i) = temp(i, 0);
            }
            
            // Normalize
            eigenvalue = v.norm();
            for (size_t i = 0; i < n; ++i) {
                v(0, i) /= eigenvalue;
            }
            
            // Check convergence
            float diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                diff += std::abs(v(0, i) - v_prev(0, i));
            }
            
            if (diff < tol) break;
        }
        
        return {eigenvalue, v};
    }
    
    void deflate(Matrix<float>& A, float eigenvalue, const Matrix<float>& eigenvector) const {
        // Outer product: v^T * v
        Matrix<float> outer(eigenvector.num_cols(), eigenvector.num_cols());
        for (size_t i = 0; i < eigenvector.num_cols(); ++i) {
            for (size_t j = 0; j < eigenvector.num_cols(); ++j) {
                outer(i, j) = eigenvector(0, i) * eigenvector(0, j) * eigenvalue;
            }
        }
        
        A = A - outer;
    }
};