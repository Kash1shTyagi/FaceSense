#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <limits>

/**
 * @brief Multivariate Gaussian and GMM (EM) in PCA feature space.
 * Template on T (float/double).
 */
namespace GMM {

    /**
     * @brief Simple struct for a single Gaussian component.
     */
    template<typename T>
    struct GaussianComponent {
        std::vector<T> mean;        // length k
        std::vector<std::vector<T>> cov;      // k x k covariance matrix
        std::vector<std::vector<T>> cov_inv;  // precomputed inverse
        T cov_det;                  // precomputed determinant
        T weight;                   // mixing weight π_m

        /**
         * @brief Initialize component with given dimension k.
         */
        GaussianComponent(size_t k = 0)
            : mean(k, T{}),
              cov(k, std::vector<T>(k, T{})),
              cov_inv(k, std::vector<T>(k, T{})),
              cov_det(T{1}),
              weight(T{1}) {}

        /**
         * @brief Compute PDF N(x; mean, cov). Assumes cov_inv and cov_det are set.
         */
        T pdf(const std::vector<T>& x) const {
            size_t k = mean.size();
            if (x.size() != k)
                throw std::runtime_error("Dimension mismatch in GaussianComponent::pdf");
            // Compute (x - mean)
            std::vector<T> diff(k);
            for (size_t i = 0; i < k; ++i)
                diff[i] = x[i] - mean[i];
            // Compute Mahalanobis: diff^T * cov_inv * diff
            std::vector<T> tmp(k, T{});
            for (size_t i = 0; i < k; ++i) {
                T acc = T{};
                for (size_t j = 0; j < k; ++j) {
                    acc += cov_inv[i][j] * diff[j];
                }
                tmp[i] = acc;
            }
            T mah = T{};
            for (size_t i = 0; i < k; ++i)
                mah += diff[i] * tmp[i];
            T norm_const = std::pow(static_cast<T>(2 * M_PI), -static_cast<T>(k) / 2) * std::pow(cov_det, -static_cast<T>(0.5));
            return norm_const * std::exp(-static_cast<T>(0.5) * mah);
        }
    };

    /**
     * @brief Compute inverse and determinant of a matrix.
     * You can implement using Gaussian elimination or Cholesky if covariances are SPD.
     * For simplicity, one may assume diagonal covariance (store only diagonal), then inverse and det are trivial.
     *
     * Here declare a utility; implement in GMM.cpp:
     * - For full cov: use naive inversion (e.g., Gaussian elimination) or better Cholesky.
     * - For diagonal: extract diagonal elements.
     */
    template<typename T>
    void invertMatrix(const std::vector<std::vector<T>>& mat,
                      std::vector<std::vector<T>>& mat_inv,
                      T& det) {
        // Implement full inversion or throw if unimplemented.
        // For now, assume diagonal case: check off-diagonals are zero.
        size_t k = mat.size();
        mat_inv.assign(k, std::vector<T>(k, T{}));
        det = T{1};
        for (size_t i = 0; i < k; ++i) {
            bool diagonal = true;
            for (size_t j = 0; j < k; ++j) {
                if (i != j && std::abs(mat[i][j]) > std::numeric_limits<T>::epsilon()) {
                    diagonal = false;
                    break;
                }
            }
            if (!diagonal) {
                throw std::runtime_error("Full covariance inversion not implemented; consider diagonal covariance");
            }
            T val = mat[i][i];
            if (val <= T{}) throw std::runtime_error("Non-positive diagonal in covariance");
            mat_inv[i][i] = T{1} / val;
            det *= val;
        }
    }

    /**
     * @brief GMM model: holds M components and performs EM training (offline) and inference.
     */
    template<typename T>
    class GMMModel {
    public:
        GMMModel(size_t num_components = 0, size_t dim = 0)
            : M(num_components), k(dim) {
            components.reserve(M);
            for (size_t m = 0; m < M; ++m)
                components.emplace_back(dim);
        }

        /**
         * @brief Initialize parameters: e.g., random means from data, equal weights, identity or diagonal cov.
         * Implement as needed.
         */
        void initialize(const std::vector<std::vector<T>>& data) {
            // e.g., random select means, weights = 1/M, cov = identity or sample variance.
            // This is user-implemented in GMM.cpp.
        }

        /**
         * @brief Fit GMM via EM on provided data.
         * @param data: N samples of length k.
         * @param max_iters: max EM iterations.
         * @param tol: tolerance on log-likelihood improvement.
         */
        void fit(const std::vector<std::vector<T>>& data, int max_iters = 100, T tol = static_cast<T>(1e-4)) {
            size_t N = data.size();
            if (N == 0) throw std::runtime_error("No data for GMM fit");
            // Allocate responsibility matrix gamma: N x M
            std::vector<std::vector<T>> gamma(N, std::vector<T>(M, T{}));
            T prev_ll = -std::numeric_limits<T>::infinity();

            // EM loop (outline)
            for (int iter = 0; iter < max_iters; ++iter) {
                // E-step: compute responsibilities
                for (size_t i = 0; i < N; ++i) {
                    T sum_resp = T{};
                    for (size_t m = 0; m < M; ++m) {
                        T p = components[m].weight * components[m].pdf(data[i]);
                        gamma[i][m] = p;
                        sum_resp += p;
                    }
                    if (sum_resp <= T{}) {
                        // underflow or zero; distribute uniformly
                        T uniform = T{1} / static_cast<T>(M);
                        for (size_t m = 0; m < M; ++m)
                            gamma[i][m] = uniform;
                    } else {
                        for (size_t m = 0; m < M; ++m)
                            gamma[i][m] /= sum_resp;
                    }
                }
                // M-step: update parameters
                for (size_t m = 0; m < M; ++m) {
                    T N_m = T{};
                    // Sum responsibilities
                    for (size_t i = 0; i < N; ++i)
                        N_m += gamma[i][m];
                    if (N_m <= T{}) N_m = std::numeric_limits<T>::epsilon();
                    // Update weight
                    components[m].weight = N_m / static_cast<T>(N);
                    // Update mean
                    std::vector<T> new_mean(k, T{});
                    for (size_t i = 0; i < N; ++i) {
                        for (size_t d = 0; d < k; ++d)
                            new_mean[d] += gamma[i][m] * data[i][d];
                    }
                    for (size_t d = 0; d < k; ++d)
                        new_mean[d] /= N_m;
                    components[m].mean = new_mean;
                    // Update covariance
                    std::vector<std::vector<T>> new_cov(k, std::vector<T>(k, T{}));
                    for (size_t i = 0; i < N; ++i) {
                        // diff = x - mean
                        std::vector<T> diff(k);
                        for (size_t d = 0; d < k; ++d)
                            diff[d] = data[i][d] - components[m].mean[d];
                        // outer product diff * diff^T
                        for (size_t u = 0; u < k; ++u) {
                            for (size_t v = 0; v < k; ++v) {
                                new_cov[u][v] += gamma[i][m] * diff[u] * diff[v];
                            }
                        }
                    }
                    for (size_t u = 0; u < k; ++u)
                        for (size_t v = 0; v < k; ++v)
                            new_cov[u][v] /= N_m;
                    components[m].cov = new_cov;
                    // Compute inverse and determinant
                    invertMatrix(components[m].cov, components[m].cov_inv, components[m].cov_det);
                }
                // Compute log-likelihood
                T ll = T{};
                for (size_t i = 0; i < N; ++i) {
                    T sum_p = T{};
                    for (size_t m = 0; m < M; ++m)
                        sum_p += components[m].weight * components[m].pdf(data[i]);
                    if (sum_p > T{})
                        ll += std::log(sum_p);
                    else
                        ll += std::log(std::numeric_limits<T>::min());
                }
                if (std::abs(ll - prev_ll) < tol)
                    break;
                prev_ll = ll;
            }
        }

        /**
         * @brief Given a new sample, compute posterior probabilities over components.
         * @param x: sample vector length k.
         * @return vector<T> of size M with normalized posterior probabilities.
         */
        std::vector<T> infer(const std::vector<T>& x) const {
            std::vector<T> post(M, T{});
            T sum_p = T{};
            for (size_t m = 0; m < M; ++m) {
                T p = components[m].weight * components[m].pdf(x);
                post[m] = p;
                sum_p += p;
            }
            if (sum_p <= T{}) {
                T uniform = T{1} / static_cast<T>(M);
                for (size_t m = 0; m < M; ++m)
                    post[m] = uniform;
            } else {
                for (size_t m = 0; m < M; ++m)
                    post[m] /= sum_p;
            }
            return post;
        }

        const std::vector<GaussianComponent<T>>& getComponents() const {
            return components;
        }

        /**
         * @brief Serialize/deserialize parameters to/from binary files.
         * Implement in Utils.cpp: write number of components M, dimension k, then for each:
         * weight, mean vector, cov matrix.
         */
        void saveToFile(const std::string& filename) const;
        void loadFromFile(const std::string& filename);

    private:
        size_t M;  // number of components
        size_t k;  // dimension
        std::vector<GaussianComponent<T>> components;
    };

} // namespace GMM
