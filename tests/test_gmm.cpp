#include "GMM.hpp"
#include <gtest/gtest.h>
#include <vector>

using G = GMM::GaussianComponent<double>;
using Model = GMM::GMMModel<double>;

TEST(GMMTest, FitTwoClusters1D) {
    // Generate 1D data: around 0 and 10
    std::vector<std::vector<double>> data;
    for (double x : { -0.1, 0.0, 0.1 }) data.push_back({x});
    for (double x : { 9.9, 10.0, 10.1 }) data.push_back({x});

    Model gmm(2, 1);
    gmm.initialize(data);
    gmm.fit(data, /*max_iters=*/50, /*tol=*/1e-4);

    auto comps = gmm.getComponents();
    ASSERT_EQ(comps.size(), 2u);

    // Extract means & sort
    std::vector<double> means = { comps[0].mean[0], comps[1].mean[0] };
    std::sort(means.begin(), means.end());

    EXPECT_NEAR(means[0], 0.0, 0.5);
    EXPECT_NEAR(means[1], 10.0, 0.5);
}

TEST(GMMTest, PosteriorSumsToOne) {
    Model gmm(3, 1);
    // Manually set up three identical components
    for (auto &comp : gmm.getComponents()) {
        comp.mean = {0.0};
        comp.cov = { {1.0} };
        GMM::invertMatrix(comp.cov, comp.cov_inv, comp.cov_det);
        comp.weight = 1.0/3.0;
    }
    auto post = gmm.infer({0.5});
    double sum = post[0] + post[1] + post[2];
    EXPECT_NEAR(sum, 1.0, 1e-6);
}
