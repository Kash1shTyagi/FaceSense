#include "PCA.hpp"

// Explicit instantiations for double
namespace PCA {
    template std::vector<double> computeMean<double>(const std::vector<std::vector<double>>&);
    template void subtractMean<double>(std::vector<std::vector<double>>&, const std::vector<double>&);
    template Matrix<double> computeCovariance<double>(const std::vector<std::vector<double>>&);
    template std::pair<double, std::vector<double>> powerMethod<double>(const Matrix<double>&, int, double);
    template void deflate<double>(Matrix<double>&, double, const std::vector<double>&);
    template std::pair<std::vector<double>, std::vector<std::vector<double>>> computeTopKEigen<double>(
        const Matrix<double>&, int, int, double);
    template std::vector<double> project<double>(const std::vector<double>&, const std::vector<std::vector<double>>&);
}
