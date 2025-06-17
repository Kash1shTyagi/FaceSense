#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <immintrin.h>
#include <random>
#include <functional>
#include <sstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class Matrix {
public:
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols) {}
    
    Matrix(size_t rows, size_t cols, const T& init) : rows(rows), cols(cols), data(rows * cols, init) {}
    
    Matrix(size_t rows, size_t cols, T* external_data) 
        : rows(rows), cols(cols), data(external_data, external_data + rows * cols) {}
    
    // Accessors
    T& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const T& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    
    // Dimensions
    size_t num_rows() const noexcept { return rows; }
    size_t num_cols() const noexcept { return cols; }
    size_t size() const noexcept { return rows * cols; }
    
    // Row operations
    void set_row(size_t i, const std::vector<T>& row) {
        if (row.size() != cols) {
            throw std::invalid_argument("Row size mismatch");
        }
        std::copy(row.begin(), row.end(), data.begin() + i * cols);
    }
    
    std::vector<T> get_row(size_t i) const {
        if (i >= rows) throw std::out_of_range("Row index out of range");
        return std::vector<T>(data.begin() + i * cols, 
                             data.begin() + (i + 1) * cols);
    }
    
    Vector<T> row_vector(size_t i) const {
        if (i >= rows) throw std::out_of_range("Row index out of range");
        Vector<T> vec(1, cols);
        for (size_t j = 0; j < cols; j++) {
            vec(0, j) = (*this)(i, j);
        }
        return vec;
    }
    
    // Matrix operations
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        Matrix result(rows, other.cols);
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < cols; k++) {
                T a = (*this)(i, k);
                for (size_t j = 0; j < other.cols; j++) {
                    result(i, j) += a * other(k, j);
                }
            }
        }
        return result;
    }
    
    Matrix operator*(T scalar) const {
        Matrix result(rows, cols);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    
    friend Matrix operator*(T scalar, const Matrix& mat) {
        return mat * scalar;
    }
    
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        Matrix result(rows, cols);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        Matrix result(rows, cols);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    
    // Vector operations
    T norm() const {
        if (rows != 1 && cols != 1) {
            throw std::runtime_error("Norm only defined for vectors");
        }
        
        T sum = 0;
        for (const auto& val : data) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
    
    // Serialization
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        save(file);
    }
    
    void save(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        os.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    }
    
    static Matrix load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        return load(file);
    }
    
    static Matrix load(std::istream& is) {
        size_t rows, cols;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        Matrix result(rows, cols);
        is.read(reinterpret_cast<char*>(result.data.data()), result.size() * sizeof(T));
        return result;
    }
    
    // Data access
    const std::vector<T>& get_data() const { return data; }
    std::vector<T>& get_data() { return data; }

private:
    size_t rows, cols;
    std::vector<T> data;
};

// Vector alias
template <typename T>
using Vector = Matrix<T>;