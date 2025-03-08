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
}

// Test the TradingModel class
// In a real test, we would use a mock model
// For this example, we'll just test the feature extraction
TEST(TradingModelTest, FeatureExtraction) {
    // Use default constructor for testing feature extraction
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
}

} // namespace 