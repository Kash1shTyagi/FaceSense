#include "GMM.hpp"
#include <fstream>

namespace GMM {

//
// Simple “diagonal-only” invertMatrix implementation
//
template<typename T>
void invertMatrix(const std::vector<std::vector<T>>& mat,
                  std::vector<std::vector<T>>& mat_inv,
                  T& det) {
    size_t k = mat.size();
    mat_inv.assign(k, std::vector<T>(k, T{}));
    det = T{1};
    for (size_t i = 0; i < k; ++i) {
        T d = mat[i][i];
        if (d <= T{}) throw std::runtime_error("Non-positive variance in diagonal GMM");
        mat_inv[i][i] = T{1} / d;
        det *= d;
    }
}

// Explicitly instantiate
template void invertMatrix<double>(
    const std::vector<std::vector<double>>&,
    std::vector<std::vector<double>>&,
    double&
);

//
// GMMModel methods: save/load parameters (binary).
//
template<typename T>
void GMMModel<T>::saveToFile(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open GMM save file");
    // Write M, k
    out.write(reinterpret_cast<const char*>(&M), sizeof(M));
    out.write(reinterpret_cast<const char*>(&k), sizeof(k));
    for (const auto& comp : components) {
        // weight
        out.write(reinterpret_cast<const char*>(&comp.weight), sizeof(comp.weight));
        // mean
        out.write(reinterpret_cast<const char*>(comp.mean.data()), comp.mean.size() * sizeof(T));
        // cov diagonal only
        for (size_t i = 0; i < k; ++i) {
            T var = comp.cov[i][i];
            out.write(reinterpret_cast<const char*>(&var), sizeof(T));
        }
    }
}

template<typename T>
void GMMModel<T>::loadFromFile(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open GMM load file");
    size_t m_read, k_read;
    in.read(reinterpret_cast<char*>(&m_read), sizeof(m_read));
    in.read(reinterpret_cast<char*>(&k_read), sizeof(k_read));
    if (m_read != M || k_read != k)
        throw std::runtime_error("GMM parameter file dimension mismatch");
    for (auto& comp : components) {
        // weight
        in.read(reinterpret_cast<char*>(&comp.weight), sizeof(comp.weight));
        // mean
        in.read(reinterpret_cast<char*>(comp.mean.data()), comp.mean.size() * sizeof(T));
        // cov diag only
        for (size_t i = 0; i < k; ++i) {
            T var;
            in.read(reinterpret_cast<char*>(&var), sizeof(T));
            comp.cov[i][i] = var;
        }
        // rebuild inv & det
        invertMatrix(comp.cov, comp.cov_inv, comp.cov_det);
    }
}

// Explicit instantiation
template class GMMModel<double>;

} // namespace GMM
