#include "PCA.hpp"
#include "Matrix.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace PCA;

// Simple 2D dataset for which covariance is diagonal [4,1]:
// Points: (3,2), (1,2), (3,0), (1,0).  
// Mean = (2,1); zero-mean data has cov_x = 4, cov_y = 1.
TEST(PCATest, ComputeMeanAndCovariance) {
    std::vector<std::vector<double>> data = {
        {3,2}, {1,2}, {3,0}, {1,0}
    };
    auto mean = computeMean(data);
    EXPECT_DOUBLE_EQ(mean[0], 2.0);
    EXPECT_DOUBLE_EQ(mean[1], 1.0);

    subtractMean(data, mean);
    auto C = computeCovariance(data); // 2×2
    // Because we didn't divide by N or N-1, covariance entries are sums:
    // C(0,0) = sum x_i^2 = (1+1+1+1)=4; C(1,1)=4 as well?
    // Actually zero-mean X coords: [1,-1,1,-1] → sum sq = 4
    // Y coords: [1,1,-1,-1] → sum sq = 4
    EXPECT_DOUBLE_EQ(C(0,0), 4.0);
    EXPECT_DOUBLE_EQ(C(1,1), 4.0);
    EXPECT_DOUBLE_EQ(C(0,1), 0.0);
    EXPECT_DOUBLE_EQ(C(1,0), 0.0);
}

TEST(PCATest, PowerMethodOnDiagonal) {
    // Build 2×2 diagonal matrix diag(5,2)
    Matrix<double> C(2,2);
    C(0,0) = 5; C(1,1) = 2;
    auto [lambda, v] = powerMethod<double>(C, 100, 1e-8);
    EXPECT_NEAR(lambda, 5.0, 1e-6);
    // eigenvector should be (1,0) or its negative
    EXPECT_NEAR(std::abs(v[0]), 1.0, 1e-6);
    EXPECT_NEAR(std::abs(v[1]), 0.0, 1e-6);
}

TEST(PCATest, DeflationAndTopKEigen) {
    // C = diag(3,1)
    Matrix<double> C(2,2);
    C(0,0)=3; C(1,1)=1;
    auto [lams, vecs] = computeTopKEigen<double>(C, 2, 100, 1e-8);
    EXPECT_EQ(lams.size(), 2);
    EXPECT_NEAR(lams[0], 3.0, 1e-6);
    EXPECT_NEAR(lams[1], 1.0, 1e-6);
    // Check that each vector is normalized
    for (auto &v : vecs) {
        double norm = std::sqrt(v[0]*v[0] + v[1]*v[1]);
        EXPECT_NEAR(norm, 1.0, 1e-6);
    }
}
