// src/GMM.cpp

#include "GMM.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace GMM {

//
// GMMModel<T> save/load
//
template<typename T>
void GMMModel<T>::saveToFile(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open GMM save file");

    // Write component count M and dimension k as 32-bit ints
    uint32_t M32 = static_cast<uint32_t>(M);
    uint32_t k32 = static_cast<uint32_t>(k);
    out.write(reinterpret_cast<const char*>(&M32), sizeof(M32));
    out.write(reinterpret_cast<const char*>(&k32), sizeof(k32));

    for (const auto& comp : components) {
        // weight
        out.write(reinterpret_cast<const char*>(&comp.weight), sizeof(comp.weight));
        // mean vector
        out.write(reinterpret_cast<const char*>(comp.mean.data()),
                  comp.mean.size() * sizeof(T));
        // covariance diagonal only
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

    // Read M and k as 32-bit ints
    uint32_t M32 = 0, k32 = 0;
    in.read(reinterpret_cast<char*>(&M32), sizeof(M32));
    in.read(reinterpret_cast<char*>(&k32), sizeof(k32));

    if (M32 != static_cast<uint32_t>(M) || k32 != static_cast<uint32_t>(k)) {
        std::ostringstream oss;
        oss << "GMM parameter file mismatch: file has M=" << M32
            << ", k=" << k32
            << " but model expects M=" << M << ", k=" << k;
        throw std::runtime_error(oss.str());
    }

    for (auto& comp : components) {
        // weight
        in.read(reinterpret_cast<char*>(&comp.weight), sizeof(comp.weight));
        // mean vector
        in.read(reinterpret_cast<char*>(comp.mean.data()),
                comp.mean.size() * sizeof(T));
        // covariance diagonal
        for (size_t i = 0; i < k; ++i) {
            T var;
            in.read(reinterpret_cast<char*>(&var), sizeof(T));
            comp.cov[i][i] = var;
        }
        // rebuild inverse and determinant via inline invertMatrix<T>
        invertMatrix(comp.cov, comp.cov_inv, comp.cov_det);
    }
}

// Explicit instantiation of GMMModel for double
template class GMMModel<double>;

} // namespace GMM
