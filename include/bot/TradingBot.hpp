#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include <functional>
#include <condition_variable>
#include <queue>
#include <nlohmann/json.hpp>

// Include the full definition of c10::IValue
#include <ATen/core/ivalue.h>

#include "orderbook/OrderbookFetcher.hpp"
#include "model/TradingModel.hpp"

namespace trading {

/**
 * @brief Represents a trade order
 */
struct TradeOrder {
    enum class Type {
        BUY,
        SELL
    };
    
    enum class Status {
        PENDING,
        FILLED,
        PARTIALLY_FILLED,
        CANCELED,
        REJECTED
    };
    
    std::string id;
    Type type;
    double price;
    double quantity;
    double filled_quantity = 0.0;
    Status status = Status::PENDING;
    std::chrono::system_clock::time_point timestamp;
    
    TradeOrder(Type t, double p, double q)
        : type(t), price(p), quantity(q), timestamp(std::chrono::system_clock::now()) {
        // Generate a unique ID
        id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    }
};

/**
 * @brief Represents the trading bot that uses orderbook data to make trading decisions
 */
class TradingBot {
public:
    TradingBot();
    ~TradingBot();
    
    /**
     * @brief Initialize the trading bot
     * @param exchange_url The URL of the exchange websocket
     * @param symbol The trading pair symbol
     * @param model_path The path to the trained model
     * @return true if initialization was successful, false otherwise
     */
    bool initialize(const std::string& exchange_url, const std::string& symbol, const std::string& model_path);
    
    /**
     * @brief Start the trading bot
     */
    void start();
    
    /**
     * @brief Stop the trading bot
     */
    void stop();
    
    /**
     * @brief Get the current position
     * @return The current position (positive for long, negative for short)
     */
    double getPosition() const;
    
    /**
     * @brief Get the current balance
     * @return The current balance
     */
    double getBalance() const;
    
    /**
     * @brief Get the list of open orders
     * @return The list of open orders
     */
    std::vector<TradeOrder> getOpenOrders() const;
    
    /**
     * @brief Get the list of filled orders
     * @return The list of filled orders
     */
    std::vector<TradeOrder> getFilledOrders() const;
    
    /**
     * @brief Get the performance metrics
     * @return The performance metrics as a JSON object
     */
    nlohmann::json getPerformanceMetrics() const;
    
private:
    void run();
    void onOrderbookUpdate(const Orderbook& orderbook);
    void processModelPrediction(double prediction);
    bool placeOrder(const TradeOrder& order);
    bool cancelOrder(const std::string& order_id);
    void updateOrderStatus();
    void calculatePerformanceMetrics();
    
    std::unique_ptr<OrderbookFetcher> orderbook_fetcher_;
    std::unique_ptr<TradingModel> trading_model_;
    
    std::atomic<bool> running_;
    std::thread trading_thread_;
    std::condition_variable cv_;
    
    double position_ = 0.0;
    double balance_ = 0.0;
    
    std::vector<TradeOrder> open_orders_;
    std::vector<TradeOrder> filled_orders_;
    mutable std::mutex orders_mutex_;
    
    // Performance metrics
    std::chrono::nanoseconds avg_tick_to_trade_;
    std::atomic<uint64_t> trades_per_second_;
    std::chrono::system_clock::time_point last_metrics_update_;
    uint64_t total_trades_;
    uint64_t successful_trades_;
    mutable std::mutex metrics_mutex_;
    
    // Configuration
    std::string symbol_;
    double max_position_ = 1.0;
    double order_size_ = 0.1;
    double min_spread_ = 0.0001;
};

} // namespace trading 