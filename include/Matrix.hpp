#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <initializer_list>

/**
 * @brief A simple dense matrix template storing elements in row-major order.
 * Template parameter T is typically double or float.
 */
template<typename T>
class Matrix {
public:
    using size_type = size_t;

    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_type rows, size_type cols, T init_val = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    // Construct from initializer list of rows: {{1,2,3}, {4,5,6}}
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        cols_ = init.begin()->size();
        data_.reserve(rows_ * cols_);
        for (auto &row : init) {
            if (row.size() != cols_)
                throw std::runtime_error("Matrix initializer rows must have same length");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // Accessors
    T& operator()(size_type i, size_type j) {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix index out of range");
        return data_[i * cols_ + j];
    }
    const T& operator()(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix index out of range");
        return data_[i * cols_ + j];
    }

    size_type rows() const { return rows_; }
    size_type cols() const { return cols_; }

    // Data pointer access (row-major)
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::runtime_error("Matrix dimensions must match for addition");
        Matrix res(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i)
            res.data_[i] = data_[i] + other.data_[i];
        return res;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::runtime_error("Matrix dimensions must match for subtraction");
        Matrix res(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i)
            res.data_[i] = data_[i] - other.data_[i];
        return res;
    }

    // Scalar multiplication
    Matrix operator*(T scalar) const {
        Matrix res(rows_, cols_);
        for (size_type i = 0; i < data_.size(); ++i)
            res.data_[i] = data_[i] * scalar;
        return res;
    }

    // Matrix multiplication
    // Note: naive O(n^3). For small D (e.g., 4096 × 4096 covariance), consider optimized or approximations.
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_)
            throw std::runtime_error("Matrix inner dimensions must match for multiplication");
        Matrix res(rows_, other.cols_, T{});
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type k = 0; k < cols_; ++k) {
                T aik = this->operator()(i, k);
                for (size_type j = 0; j < other.cols_; ++j) {
                    res(i, j) += aik * other(k, j);
                }
            }
        }
        return res;
    }

    // Transpose
    Matrix transpose() const {
        Matrix res(cols_, rows_);
        for (size_type i = 0; i < rows_; ++i)
            for (size_type j = 0; j < cols_; ++j)
                res(j, i) = this->operator()(i, j);
        return res;
    }

    // Frobenius norm
    T norm() const {
        T sum = T{};
        for (auto &v : data_)
            sum += v * v;
        return std::sqrt(sum);
    }

    // Multiply matrix by vector (vector as std::vector<T>)
    std::vector<T> mulVector(const std::vector<T>& vec) const {
        if (cols_ != vec.size())
            throw std::runtime_error("Matrix-vector dimension mismatch");
        std::vector<T> result(rows_, T{});
        for (size_type i = 0; i < rows_; ++i) {
            T acc = T{};
            for (size_type j = 0; j < cols_; ++j)
                acc += this->operator()(i, j) * vec[j];
            result[i] = acc;
        }
        return result;
    }

private:
    size_type rows_, cols_;
    std::vector<T> data_;
};
