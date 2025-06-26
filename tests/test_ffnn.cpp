#include "FFNN.hpp"
#include "Utils.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <filesystem>

using NetD = FFNN::Network<double>;

// Helper: write a small binary file of doubles
static void writeBin(const std::string& name, const std::vector<double>& v) {
    ASSERT_TRUE(Utils::writeBinary(name, v)) << "Failed to write " << name;
}

TEST(FFNNTest, ForwardIdentityWeights) {
    // dimensions
    const size_t k = 3, H = 3, C = 2;
    NetD net;
    net.init(k, H, C);

    // Prepare weight/bias vectors:
    // W1: H×k identity block in first 3 entries, rest zero
    std::vector<double> W1(H*k, 0.0);
    for (size_t i = 0; i < H && i < k; ++i)
        W1[i*k + i] = 1.0;
    std::vector<double> b1(H, 0.0);

    // W2: C×H: first row picks hidden[0], second picks hidden[1]
    std::vector<double> W2(C*H, 0.0);
    W2[0*H + 0] = 1.0;  // output0 = h0
    W2[1*H + 1] = 1.0;  // output1 = h1
    std::vector<double> b2(C, 0.0);

    // Write binary files
    writeBin("w1.bin", W1);
    writeBin("b1.bin", b1);
    writeBin("w2.bin", W2);
    writeBin("b2.bin", b2);

    // Load into network
    ASSERT_TRUE(net.loadParameters("w1.bin","b1.bin","w2.bin","b2.bin"));

    // Input vector
    std::vector<double> x = { 0.5, -1.0, 2.0 };  
    // hidden = ReLU([0.5, -1.0, 2.0]) = [0.5, 0, 2.0]
    // logits = [h0, h1] = [0.5, 0]
    // softmax → [1.0, 0.0]
    auto probs = net.forward(x);
    EXPECT_NEAR(probs[0], 1.0, 1e-6);
    EXPECT_NEAR(probs[1], 0.0, 1e-6);
}
