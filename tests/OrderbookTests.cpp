#include <gtest/gtest.h>
#include "orderbook/OrderbookFetcher.hpp"

namespace {

using namespace trading;

// Test the Orderbook class
TEST(OrderbookTest, BasicFunctionality) {
    Orderbook orderbook;
    
    // Create some test orders
    std::vector<Order> bids = {
        Order(100.0, 1.0),
        Order(101.0, 2.0),
        Order(99.0, 3.0)
    };
    
    std::vector<Order> asks = {
        Order(102.0, 1.0),
        Order(103.0, 2.0),
        Order(101.5, 3.0)
    };
    
    // Update the orderbook
    orderbook.updateBids(bids);
    orderbook.updateAsks(asks);
    
    // Check that the bids are sorted in descending order by price
    ASSERT_EQ(orderbook.getBids().size(), 3);
    EXPECT_EQ(orderbook.getBids()[0].price, 101.0);
    EXPECT_EQ(orderbook.getBids()[1].price, 100.0);
    EXPECT_EQ(orderbook.getBids()[2].price, 99.0);
    
    // Check that the asks are sorted in ascending order by price
    ASSERT_EQ(orderbook.getAsks().size(), 3);
    EXPECT_EQ(orderbook.getAsks()[0].price, 101.5);
    EXPECT_EQ(orderbook.getAsks()[1].price, 102.0);
    EXPECT_EQ(orderbook.getAsks()[2].price, 103.0);
    
    // Check the best bid and ask
    EXPECT_EQ(orderbook.getBestBid(), 101.0);
    EXPECT_EQ(orderbook.getBestAsk(), 101.5);
    
    // Check the mid price and spread
    EXPECT_EQ(orderbook.getMidPrice(), (101.0 + 101.5) / 2.0);
    EXPECT_EQ(orderbook.getSpread(), 101.5 - 101.0);
}

// Test the OrderbookFetcher class with a mock websocket
// In a real test, we would use a mock websocket server
// For this example, we'll just test the basic functionality
TEST(OrderbookFetcherTest, BasicFunctionality) {
    OrderbookFetcher fetcher;
    
    // Test the callback registration
    bool callback_called = false;
    fetcher.registerCallback([&callback_called](const Orderbook& orderbook) {
        callback_called = true;
    });
    
    // Get the latest orderbook (should be empty)
    Orderbook orderbook = fetcher.getLatestOrderbook();
    EXPECT_EQ(orderbook.getBids().size(), 0);
    EXPECT_EQ(orderbook.getAsks().size(), 0);
    
    // In a real test, we would connect to a mock websocket server
    // and test the orderbook updates
    // For this example, we'll just test that the callback registration works
    EXPECT_FALSE(callback_called);
}

} // namespace 