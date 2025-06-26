#include "Matrix.hpp"
#include <gtest/gtest.h>>

using MatrixD = Matrix<double>;

TEST(MatrixTest, AdditionAndSubtraction) {
    MatrixD A{{1.0, 2.0}, {3.0, 4.0}};
    MatrixD B{{4.0, 3.0}, {2.0, 1.0}};
    auto C = A + B;
    EXPECT_DOUBLE_EQ(C(0,0), 5.0);
    EXPECT_DOUBLE_EQ(C(0,1), 5.0);
    EXPECT_DOUBLE_EQ(C(1,0), 5.0);
    EXPECT_DOUBLE_EQ(C(1,1), 5.0);

    auto D = C - A;
    EXPECT_DOUBLE_EQ(D(0,0), 4.0);
    EXPECT_DOUBLE_EQ(D(1,1), 1.0);
}

TEST(MatrixTest, Multiplication) {
    MatrixD A{{1.0, 2.0, 3.0},
              {4.0, 5.0, 6.0}};
    MatrixD B{{7.0,  8.0},
              {9.0, 10.0},
              {11.0,12.0}};
    auto C = A * B; // 2x2
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_DOUBLE_EQ(C(0,0),  58.0); // 1*7 +2*9+3*11
    EXPECT_DOUBLE_EQ(C(0,1),  64.0);
    EXPECT_DOUBLE_EQ(C(1,0), 139.0);
    EXPECT_DOUBLE_EQ(C(1,1), 154.0);
}

TEST(MatrixTest, Transpose) {
    MatrixD A{{1,2,3},{4,5,6}};
    auto At = A.transpose();
    EXPECT_EQ(At.rows(), 3);
    EXPECT_EQ(At.cols(), 2);
    EXPECT_DOUBLE_EQ(At(2,1), 6.0);
}

TEST(MatrixTest, FrobeniusNorm) {
    MatrixD A{{3,4}};
    // norm = sqrt(3^2 + 4^2) = 5
    EXPECT_DOUBLE_EQ(A.norm(), 5.0);
}
