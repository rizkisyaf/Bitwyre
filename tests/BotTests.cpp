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
    // Create mock objects
    auto fetcher = std::make_shared<trading::OrderbookFetcher>("btcusdt");
    auto model = std::make_shared<trading::TradingModel>();
    
    // Create the bot with API keys (using dummy values for testing)
    double initial_balance = 100.0;
    std::string api_key = "dummy_api_key";
    std::string secret_key = "dummy_secret_key";
    trading::TradingBot bot(fetcher, model, initial_balance, api_key, secret_key);
    
    // Check the initial state
    EXPECT_EQ(bot.getPosition(), 0.0);
    EXPECT_EQ(bot.getBalance(), initial_balance);
    EXPECT_EQ(bot.getOpenOrders().size(), 0);
    EXPECT_EQ(bot.getFilledOrders().size(), 0);
    
    // Check the P&L metrics
    nlohmann::json pnl_metrics = bot.getPnLMetrics();
    EXPECT_EQ(pnl_metrics["realized_pnl"], 0.0);
    EXPECT_EQ(pnl_metrics["unrealized_pnl"], 0.0);
    EXPECT_EQ(pnl_metrics["total_pnl"], 0.0);
    
    // In a real test, we would initialize the bot with mock objects
    // and test the trading logic
    // For this example, we'll just test the basic functionality
}

} // namespace 