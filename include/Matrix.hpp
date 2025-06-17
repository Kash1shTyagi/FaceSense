#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <immintrin.h> 

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class AlignedAllocator {
public:
    using value_type = T;
    
    AlignedAllocator(size_t alignment = 64) : alignment_(alignment) {
        if (alignment_ < alignof(void*)) {
            alignment_ = alignof(void*);
        }
    }
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U>& other) noexcept 
        : alignment_(other.alignment) {}
    
    T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* ptr = aligned_alloc(alignment_, n * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_t) noexcept {
        std::free(ptr);
    }
    
    size_t alignment_;
};

template <typename T, size_t Alignment = 64>
class Matrix {
public:
    using value_type = T;
    using allocator_type = AlignedAllocator<T>;
    
    Matrix() : rows_(0), cols_(0) {}
    
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), 
          data_(allocator_type().allocate(rows * cols), rows_ * cols) {}
    
    Matrix(size_t rows, size_t cols, const T& value)
        : Matrix(rows, cols) {
        std::fill_n(data_.get(), size(), value);
    }
    
    // Rule of five
    Matrix(const Matrix& other) : Matrix(other.rows_, other.cols_) {
        std::copy(other.data_.get(), other.data_.get() + size(), data_.get());
    }
    
    Matrix(Matrix&& other) noexcept 
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix temp(other);
            swap(temp);
        }
        return *this;
    }
    
    Matrix& operator=(Matrix&& other) noexcept {
        swap(other);
        return *this;
    }
    
    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(data_, other.data_);
    }
    
    // Accessors
    T& operator()(size_t i, size_t j) noexcept { 
        return data_[i * cols_ + j]; 
    }
    
    const T& operator()(size_t i, size_t j) const noexcept { 
        return data_[i * cols_ + j]; 
    }
    
    // Dimensions
    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    
    // Matrix operations
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        Matrix result(rows_, other.cols_);
        
        if constexpr (std::is_same_v<T, float> && USE_AVX) {
            avx_matrix_multiply(other, result);
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t k = 0; k < cols_; ++k) {
                    T a = (*this)(i, k);
                    for (size_t j = 0; j < other.cols_; ++j) {
                        result(i, j) += a * other(k, j);
                    }
                }
            }
        }
        return result;
    }
    
    // Serialization
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        file.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
        file.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
        file.write(reinterpret_cast<const char*>(data_.get()), size() * sizeof(T));
    }
    
    static Matrix load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        Matrix result(rows, cols);
        file.read(reinterpret_cast<char*>(result.data_.get()), result.size() * sizeof(T));
        return result;
    }
    
private:
    size_t rows_, cols_;
    std::unique_ptr<T[], allocator_type> data_;
    
    // AVX-optimized matrix multiplication (float only)
    void avx_matrix_multiply(const Matrix& other, Matrix& result) const {
        static_assert(std::is_same_v<T, float>, "AVX only for float");
        
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                __m256 a = _mm256_set1_ps((*this)(i, k));
                
                size_t j = 0;
                for (; j + 8 <= other.cols_; j += 8) {
                    __m256 b = _mm256_load_ps(&other(k, j));
                    __m256 c = _mm256_load_ps(&result(i, j));
                    c = _mm256_fmadd_ps(a, b, c);
                    _mm256_store_ps(&result(i, j), c);
                }
                
                // Remainder
                for (; j < other.cols_; ++j) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
    }
};


template <typename T>
using Vector = Matrix<T, 1>;