#include <gtest/gtest.h>
#include "model/TradingModel.hpp"

namespace {

using namespace trading;

// Test the FeatureVector struct
TEST(FeatureVectorTest, BasicFunctionality) {
    FeatureVector features;
    
    // Set some basic features
    features.mid_price = 100.0;
    features.spread = 1.0;
    
    // Set bid prices and quantities
    features.bid_prices = {99.0, 98.0, 97.0};
    features.bid_quantities = {1.0, 2.0, 3.0};
    
    // Set ask prices and quantities
    features.ask_prices = {101.0, 102.0, 103.0};
    features.ask_quantities = {1.0, 2.0, 3.0};
    
    // Set additional features
    features.bid_ask_imbalance = 0.0;
    features.volume_weighted_mid_price = 100.0;
    features.price_momentum = 0.01;
    
    // Check the features
    EXPECT_EQ(features.mid_price, 100.0);
    EXPECT_EQ(features.spread, 1.0);
    ASSERT_EQ(features.bid_prices.size(), 3);
    EXPECT_EQ(features.bid_prices[0], 99.0);
    ASSERT_EQ(features.bid_quantities.size(), 3);
    EXPECT_EQ(features.bid_quantities[0], 1.0);
    ASSERT_EQ(features.ask_prices.size(), 3);
    EXPECT_EQ(features.ask_prices[0], 101.0);
    ASSERT_EQ(features.ask_quantities.size(), 3);
    EXPECT_EQ(features.ask_quantities[0], 1.0);
    EXPECT_EQ(features.bid_ask_imbalance, 0.0);
    EXPECT_EQ(features.volume_weighted_mid_price, 100.0);
    EXPECT_EQ(features.price_momentum, 0.01);
}

// Test the TradingModel class
// In a real test, we would use a mock model
// For this example, we'll just test the feature extraction
TEST(TradingModelTest, FeatureExtraction) {
    TradingModel model;
    
    // Create a test orderbook
    Orderbook orderbook;
    
    // Create some test orders
    std::vector<Order> bids = {
        Order(99.0, 1.0),
        Order(98.0, 2.0),
        Order(97.0, 3.0)
    };
    
    std::vector<Order> asks = {
        Order(101.0, 1.0),
        Order(102.0, 2.0),
        Order(103.0, 3.0)
    };
    
    // Update the orderbook
    orderbook.updateBids(bids);
    orderbook.updateAsks(asks);
    
    // Extract features
    FeatureVector features = model.extractFeatures(orderbook);
    
    // Check the basic features
    EXPECT_EQ(features.mid_price, 100.0);
    EXPECT_EQ(features.spread, 2.0);
    
    // Check the bid prices and quantities
    ASSERT_GE(features.bid_prices.size(), 3);
    EXPECT_EQ(features.bid_prices[0], 99.0);
    EXPECT_EQ(features.bid_prices[1], 98.0);
    EXPECT_EQ(features.bid_prices[2], 97.0);
    ASSERT_GE(features.bid_quantities.size(), 3);
    EXPECT_EQ(features.bid_quantities[0], 1.0);
    EXPECT_EQ(features.bid_quantities[1], 2.0);
    EXPECT_EQ(features.bid_quantities[2], 3.0);
    
    // Check the ask prices and quantities
    ASSERT_GE(features.ask_prices.size(), 3);
    EXPECT_EQ(features.ask_prices[0], 101.0);
    EXPECT_EQ(features.ask_prices[1], 102.0);
    EXPECT_EQ(features.ask_prices[2], 103.0);
    ASSERT_GE(features.ask_quantities.size(), 3);
    EXPECT_EQ(features.ask_quantities[0], 1.0);
    EXPECT_EQ(features.ask_quantities[1], 2.0);
    EXPECT_EQ(features.ask_quantities[2], 3.0);
    
    // Check the additional features
    // The exact values depend on the implementation, so we'll just check that they're reasonable
    EXPECT_GE(features.bid_ask_imbalance, -1.0);
    EXPECT_LE(features.bid_ask_imbalance, 1.0);
    EXPECT_GT(features.volume_weighted_mid_price, 0.0);
    
    // Convert features to tensor
    torch::Tensor tensor = model.featuresToTensor(features);
    
    // Check the tensor shape
    EXPECT_EQ(tensor.sizes()[0], 1);
    EXPECT_GT(tensor.sizes()[1], 0);
}

} // namespace 