#pragma once
#include "Matrix.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <random>
#include <algorithm>
#include <fstream>

class GMM {
public:
    struct Component {
        Vector<float> mean;
        Vector<float> variances;  // Diagonal covariance
        float weight;
        
        float pdf(const Vector<float>& x) const {
            size_t dim = mean.num_cols();
            float exponent = 0.0f;
            
            for (size_t i = 0; i < dim; ++i) {
                float diff = x(0, i) - mean(0, i);
                exponent += diff * diff / variances(0, i);
            }
            
            float normalization = 1.0f;
            for (size_t i = 0; i < dim; ++i) {
                normalization *= 2 * M_PI * variances(0, i);
            }
            normalization = std::sqrt(normalization);
            
            return std::exp(-0.5f * exponent) / normalization;
        }
        
        void save(std::ostream& os) const {
            mean.save(os);
            variances.save(os);
            os.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
        
        void load(std::istream& is) {
            mean = Vector<float>::load(is);
            variances = Vector<float>::load(is);
            is.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
    };
    
    GMM() = default;
    
    void train(const Matrix<float>& data, size_t components, 
               size_t max_iter = 100, float tol = 1e-6f) {
        size_t n = data.num_rows();
        size_t dim = data.num_cols();
        
        initialize_components(data, components);
        Matrix<float> responsibilities(n, components, 0.0f);
        float prev_log_likelihood = -std::numeric_limits<float>::max();
        
        for (size_t iter = 0; iter < max_iter; ++iter) {
            // E-step: Compute responsibilities
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (size_t i = 0; i < n; ++i) {
                Vector<float> sample = data.row_vector(i);
                float sum_prob = 0.0f;
                
                for (size_t k = 0; k < components; ++k) {
                    responsibilities(i, k) = components_[k].weight * 
                                            components_[k].pdf(sample);
                    sum_prob += responsibilities(i, k);
                }
                
                if (sum_prob > 0) {
                    for (size_t k = 0; k < components; ++k) {
                        responsibilities(i, k) /= sum_prob;
                    }
                }
            }
            
            // M-step: Update parameters
            update_parameters(data, responsibilities);
            
            // Check convergence
            float log_likelihood = compute_log_likelihood(data);
            if (std::abs(log_likelihood - prev_log_likelihood) < tol) break;
            prev_log_likelihood = log_likelihood;
        }
    }
    
    std::vector<float> predict(const Vector<float>& x) const {
        std::vector<float> probs(components_.size());
        float sum = 0.0f;
        
        for (size_t k = 0; k < components_.size(); ++k) {
            probs[k] = components_[k].weight * components_[k].pdf(x);
            sum += probs[k];
        }
        
        if (sum > 0) {
            for (float& p : probs) p /= sum;
        }
        
        return probs;
    }
    
    // Serialization
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        size_t num_components = components_.size();
        file.write(reinterpret_cast<const char*>(&num_components), sizeof(num_components));
        
        for (const auto& comp : components_) {
            comp.save(file);
        }
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        size_t num_components;
        file.read(reinterpret_cast<char*>(&num_components), sizeof(num_components));
        components_.resize(num_components);
        
        for (auto& comp : components_) {
            comp.load(file);
        }
    }
    
private:
    std::vector<Component> components_;
    
    void initialize_components(const Matrix<float>& data, size_t components) {
        size_t n = data.num_rows();
        size_t dim = data.num_cols();
        components_.resize(components);
        
        // K-means++ initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        
        // First centroid
        size_t first_idx = dist(gen);
        components_[0].mean = data.row_vector(first_idx);
        
        // Subsequent centroids
        for (size_t k = 1; k < components; ++k) {
            Vector<float> distances(1, n, 0.0f);
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                Vector<float> sample = data.row_vector(i);
                
                for (size_t j = 0; j < k; ++j) {
                    float dist = (sample - components_[j].mean).norm();
                    if (dist < min_dist) min_dist = dist;
                }
                
                distances(0, i) = min_dist * min_dist;
            }
            
            // Select proportional to distance squared
            std::discrete_distribution<size_t> idx_dist(
                distances.get_data().begin(), distances.get_data().end());
            size_t new_idx = idx_dist(gen);
            components_[k].mean = data.row_vector(new_idx);
        }
        
        // Initialize variances and weights
        for (auto& comp : components_) {
            comp.variances = Vector<float>(1, dim, 1.0f);  // Initial variance
            comp.weight = 1.0f / components;
        }
    }
    
    void update_parameters(const Matrix<float>& data, 
                          const Matrix<float>& responsibilities) {
        size_t n = data.num_rows();
        size_t dim = data.num_cols();
        size_t components = components_.size();
        
        // Update weights and means
        for (size_t k = 0; k < components; ++k) {
            float sum_resp = 0.0f;
            Vector<float> new_mean(1, dim, 0.0f);
            
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:sum_resp)
            #endif
            for (size_t i = 0; i < n; ++i) {
                float resp = responsibilities(i, k);
                sum_resp += resp;
                new_mean = new_mean + data.row_vector(i) * resp;
            }
            
            components_[k].weight = sum_resp / n;
            if (sum_resp > 0) {
                components_[k].mean = new_mean * (1.0f / sum_resp);
            }
        }
        
        // Update variances (diagonal covariance)
        for (size_t k = 0; k < components; ++k) {
            Vector<float> new_var(1, dim, 0.0f);
            float sum_resp = 0.0f;
            
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:sum_resp)
            #endif
            for (size_t i = 0; i < n; ++i) {
                float resp = responsibilities(i, k);
                Vector<float> diff = data.row_vector(i) - components_[k].mean;
                
                for (size_t j = 0; j < dim; ++j) {
                    new_var(0, j) += resp * diff(0, j) * diff(0, j);
                }
                sum_resp += resp;
            }
            
            if (sum_resp > 0) {
                components_[k].variances = new_var * (1.0f / sum_resp);
                
                // Add regularization to avoid zero variance
                for (size_t j = 0; j < dim; ++j) {
                    components_[k].variances(0, j) = 
                        std::max(components_[k].variances(0, j), 1e-6f);
                }
            }
        }
    }
    
    float compute_log_likelihood(const Matrix<float>& data) const {
        float log_likelihood = 0.0f;
        size_t n = data.num_rows();
        
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:log_likelihood)
        #endif
        for (size_t i = 0; i < n; ++i) {
            float sample_prob = 0.0f;
            Vector<float> sample = data.row_vector(i);
            
            for (const auto& comp : components_) {
                sample_prob += comp.weight * comp.pdf(sample);
            }
            
            if (sample_prob > 0) {
                log_likelihood += std::log(sample_prob);
            }
        }
        
        return log_likelihood;
    }
};