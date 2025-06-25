#pragma once
#include "Matrix.hpp"
#include <vector>
#include <string>

/**
 * @brief PCA routines: compute mean face, covariance, eigenvectors via Power Method + deflation, projection.
 * Template on T (float/double).
 */
namespace PCA {

    /**
     * @brief Compute mean vector from a set of flattened face vectors.
     * @param data: vector of N samples, each is std::vector<T> of size D.
     * @return std::vector<T> of length D containing the mean.
     */
    template<typename T>
    std::vector<T> computeMean(const std::vector<std::vector<T>>& data) {
        if (data.empty())
            return {};
        size_t N = data.size();
        size_t D = data[0].size();
        std::vector<T> mean(D, T{});
        for (const auto& x : data) {
            if (x.size() != D)
                throw std::runtime_error("All samples must have same dimension");
            for (size_t i = 0; i < D; ++i)
                mean[i] += x[i];
        }
        for (size_t i = 0; i < D; ++i)
            mean[i] /= static_cast<T>(N);
        return mean;
    }

    /**
     * @brief Subtract mean from each sample: zero-mean data.
     */
    template<typename T>
    void subtractMean(std::vector<std::vector<T>>& data, const std::vector<T>& mean) {
        size_t D = mean.size();
        for (auto& x : data) {
            if (x.size() != D)
                throw std::runtime_error("Dimension mismatch in subtractMean");
            for (size_t i = 0; i < D; ++i)
                x[i] -= mean[i];
        }
    }

    /**
     * @brief Compute covariance matrix C = Aᵀ * A, where A is N×D zero-mean data.
     * @param data: zero-mean samples, vector of size N, each vector<T> size D.
     * @return Matrix<T> of size D×D
     *
     * Note: For large D this is expensive. In practice for face images D=4096, N~200-300,
     * one often computes smaller N×N covariance and maps eigenvectors back; but here we do direct for clarity.
     */
    template<typename T>
    Matrix<T> computeCovariance(const std::vector<std::vector<T>>& data) {
        size_t N = data.size();
        if (N == 0) throw std::runtime_error("No data for covariance");
        size_t D = data[0].size();
        // Create A matrix: N x D, but we compute C = A^T * A directly
        Matrix<T> C(D, D, T{});
        // C(i,j) = sum over samples x: x[i] * x[j]
        for (size_t n = 0; n < N; ++n) {
            const auto& x = data[n];
            if (x.size() != D) throw std::runtime_error("Inconsistent sample dimension");
            for (size_t i = 0; i < D; ++i) {
                T xi = x[i];
                for (size_t j = 0; j < D; ++j) {
                    C(i, j) += xi * x[j];
                }
            }
        }
        // Optionally divide by (N-1) or N; up to convention. For eigenvectors, scaling doesn’t matter.
        // For numerical stability, you may subtract mean earlier.
        return C;
    }

    /**
     * @brief Power Method to approximate the largest eigenvalue and eigenvector of matrix C.
     * @param C: square matrix D×D.
     * @param num_iters: maximum iterations.
     * @param tol: convergence tolerance on change in eigenvalue or eigenvector norm.
     * @return pair<eigenvalue, eigenvector> where eigenvector is std::vector<T> size D (normalized).
     */
    template<typename T>
    std::pair<T, std::vector<T>> powerMethod(const Matrix<T>& C, int num_iters = 1000, T tol = static_cast<T>(1e-6)) {
        size_t D = C.rows();
        if (C.rows() != C.cols())
            throw std::runtime_error("Matrix must be square for powerMethod");
        // Initialize v randomly (or all ones)
        std::vector<T> v(D, T{});
        for (size_t i = 0; i < D; ++i)
            v[i] = static_cast<T>(1.0);  // simpler; consider random for large dims
        // normalize
        T norm = T{};
        for (auto& vi : v) norm += vi * vi;
        norm = std::sqrt(norm);
        for (auto& vi : v) vi /= norm;

        std::vector<T> Cv(D);
        T lambda_old = T{};
        for (int iter = 0; iter < num_iters; ++iter) {
            // Compute Cv = C * v
            for (size_t i = 0; i < D; ++i) {
                T sum = T{};
                for (size_t j = 0; j < D; ++j) {
                    sum += C(i, j) * v[j];
                }
                Cv[i] = sum;
            }
            // Compute norm of Cv
            T normCv = T{};
            for (auto& cvi : Cv) normCv += cvi * cvi;
            normCv = std::sqrt(normCv);
            if (normCv == T{}) break;
            // Normalize to get next v
            for (size_t i = 0; i < D; ++i)
                v[i] = Cv[i] / normCv;
            // Rayleigh quotient as eigenvalue estimate: vᵀ(Cv)
            T lambda = T{};
            for (size_t i = 0; i < D; ++i)
                lambda += v[i] * Cv[i];
            if (std::abs(lambda - lambda_old) < tol) {
                return {lambda, v};
            }
            lambda_old = lambda;
        }
        // Final eigenvalue estimate
        // Compute C * v one more time
        for (size_t i = 0; i < D; ++i) {
            T sum = T{};
            for (size_t j = 0; j < D; ++j)
                sum += C(i, j) * v[j];
            Cv[i] = sum;
        }
        T lambda = T{};
        for (size_t i = 0; i < D; ++i)
            lambda += v[i] * Cv[i];
        return {lambda, v};
    }

    /**
     * @brief Deflate matrix C by subtracting λ * v * v^T: C_new = C - λ * (v outer v).
     * @param C: input/output matrix (square). It will be modified in place.
     * @param eigenvalue: λ.
     * @param eigenvector: v (size D), assumed normalized.
     */
    template<typename T>
    void deflate(Matrix<T>& C, T eigenvalue, const std::vector<T>& eigenvector) {
        size_t D = C.rows();
        if (C.rows() != C.cols() || eigenvector.size() != D)
            throw std::runtime_error("Dimension mismatch in deflate");
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                C(i, j) -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    /**
     * @brief Compute top-k eigenvectors (and eigenvalues) of covariance matrix C.
     * @param C_orig: input covariance matrix; will be copied internally.
     * @param k: number of components.
     * @param num_iters: iterations for powerMethod each eigen.
     * @param tol: tolerance for powerMethod.
     * @return pair of vectors: eigenvalues (size k) and eigenvectors (vector of size k, each vector<T> length D).
     */
    template<typename T>
    std::pair<std::vector<T>, std::vector<std::vector<T>>> computeTopKEigen(const Matrix<T>& C_orig, int k, int num_iters = 1000, T tol = static_cast<T>(1e-6)) {
        Matrix<T> C = C_orig;  // copy
        size_t D = C.rows();
        std::vector<T> eigenvalues;
        std::vector<std::vector<T>> eigenvectors;
        for (int i = 0; i < k; ++i) {
            auto [lambda, v] = powerMethod(C, num_iters, tol);
            eigenvalues.push_back(lambda);
            eigenvectors.push_back(v);
            deflate(C, lambda, v);
        }
        return {eigenvalues, eigenvectors};
    }

    /**
     * @brief Project a zero-mean sample x (length D) into PCA space given eigenvectors.
     * @param x_zero_mean: sample with mean subtracted.
     * @param eigenvectors: vector of k eigenvectors, each length D.
     * @return vector of length k (coefficients).
     */
    template<typename T>
    std::vector<T> project(const std::vector<T>& x_zero_mean, const std::vector<std::vector<T>>& eigenvectors) {
        size_t D = x_zero_mean.size();
        size_t k = eigenvectors.size();
        std::vector<T> alpha(k, T{});
        for (size_t j = 0; j < k; ++j) {
            if (eigenvectors[j].size() != D)
                throw std::runtime_error("Eigenvector dimension mismatch in projection");
            T dot = T{};
            for (size_t i = 0; i < D; ++i)
                dot += eigenvectors[j][i] * x_zero_mean[i];
            alpha[j] = dot;
        }
        return alpha;
    }

    /**
     * @brief Save or load functions (optional): You can define routines to write mean vector and eigenvectors to binary files.
     * Implement in Utils.cpp, e.g., writeVectorToBin, readVectorFromBin, etc.
     */

} // namespace PCA
