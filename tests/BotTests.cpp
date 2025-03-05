#include <gtest/gtest.h>
#include "bot/TradingBot.hpp"

namespace {

using namespace trading;

// Test the TradeOrder class
TEST(TradeOrderTest, BasicFunctionality) {
    // Create a buy order
    TradeOrder buy_order(TradeOrder::Type::BUY, 100.0, 1.0);
    EXPECT_EQ(buy_order.type, TradeOrder::Type::BUY);
    EXPECT_EQ(buy_order.price, 100.0);
    EXPECT_EQ(buy_order.quantity, 1.0);
    EXPECT_EQ(buy_order.filled_quantity, 0.0);
    EXPECT_EQ(buy_order.status, TradeOrder::Status::PENDING);
    
    // Create a sell order
    TradeOrder sell_order(TradeOrder::Type::SELL, 101.0, 2.0);
    EXPECT_EQ(sell_order.type, TradeOrder::Type::SELL);
    EXPECT_EQ(sell_order.price, 101.0);
    EXPECT_EQ(sell_order.quantity, 2.0);
    EXPECT_EQ(sell_order.filled_quantity, 0.0);
    EXPECT_EQ(sell_order.status, TradeOrder::Status::PENDING);
    
    // Check that the IDs are different
    EXPECT_NE(buy_order.id, sell_order.id);
}

// Test the TradingBot class
// In a real test, we would use mock objects for the OrderbookFetcher and TradingModel
// For this example, we'll just test the basic functionality
TEST(TradingBotTest, BasicFunctionality) {
    TradingBot bot;
    
    // Check the initial state
    EXPECT_EQ(bot.getPosition(), 0.0);
    EXPECT_EQ(bot.getBalance(), 0.0);
    EXPECT_EQ(bot.getOpenOrders().size(), 0);
    EXPECT_EQ(bot.getFilledOrders().size(), 0);
    
    // Check the performance metrics
    nlohmann::json metrics = bot.getPerformanceMetrics();
    EXPECT_EQ(metrics["total_trades"], 0);
    EXPECT_EQ(metrics["successful_trades"], 0);
    EXPECT_EQ(metrics["success_rate"], 0.0);
    
    // In a real test, we would initialize the bot with mock objects
    // and test the trading logic
    // For this example, we'll just test the basic functionality
}

} // namespace 