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

// Use alignas for better memory alignment and cache efficiency
struct alignas(16) TradeOrder {
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
    
    // Add entry price for P&L calculation
    double entry_price = 0.0;
    
    TradeOrder(Type t, double p, double q)
        : type(t), price(p), quantity(q), timestamp(std::chrono::system_clock::now()) {
        // Generate a unique ID
        id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        // Set entry price
        entry_price = p;
    }
};

/**
 * @brief Trading bot that uses a model to make trading decisions
 */
class TradingBot {
public:
    TradingBot(std::shared_ptr<OrderbookFetcher> fetcher, 
               std::shared_ptr<TradingModel> model,
               double initial_balance = 0.0);
    ~TradingBot();
    
    /**
     * @brief Initialize the trading bot
     * 
     * @param exchange_url The exchange websocket URL
     * @param symbol The trading symbol
     * @param model_path The path to the model file
     * @return true if initialization was successful
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
     * 
     * @return The current position
     */
    double getPosition() const;
    
    /**
     * @brief Get the current balance
     * 
     * @return The current balance
     */
    double getBalance() const;
    
    /**
     * @brief Get all open orders
     * 
     * @return A vector of open orders
     */
    std::vector<TradeOrder> getOpenOrders() const;
    
    /**
     * @brief Get all filled orders
     * 
     * @return A vector of filled orders
     */
    std::vector<TradeOrder> getFilledOrders() const;
    
    /**
     * @brief Get performance metrics
     * 
     * @return Performance metrics as JSON
     */
    nlohmann::json getPerformanceMetrics() const;
    
    /**
     * @brief Get P&L metrics
     * 
     * @return P&L metrics as JSON
     */
    nlohmann::json getPnLMetrics() const;
    
    // Risk management settings
    void setStopLossPercentage(double percentage);
    void setMaxDrawdownPercentage(double percentage);
    
private:
    void run();
    void onOrderbookUpdate(const Orderbook& orderbook);
    void processModelPrediction(double prediction, const Orderbook& orderbook_snapshot);
    bool placeOrder(const TradeOrder& order);
    bool cancelOrder(const std::string& order_id);
    void updateOrderStatus();
    void calculatePerformanceMetrics();
    
    // New method to calculate market volatility
    double calculateMarketVolatility() const;
    
    // New method to calculate P&L
    void calculatePnL(const TradeOrder& filled_order, double current_price);
    
    // Risk management parameters
    double stop_loss_percentage_ = 0.5;  // Default 0.5% stop loss
    double max_drawdown_percentage_ = 5.0;  // Default 5% max drawdown
    bool stop_loss_triggered_ = false;
    
    // Check if stop loss or max drawdown has been reached
    bool checkRiskLimits();
    
    std::shared_ptr<OrderbookFetcher> orderbook_fetcher_;
    std::shared_ptr<TradingModel> trading_model_;
    
    std::atomic<bool> running_{false};
    std::thread trading_thread_;
    std::condition_variable cv_;
    std::mutex mutex_;
    
    double position_ = 0.0;
    double balance_ = 0.0;
    double initial_balance_ = 0.0;
    
    // P&L tracking
    double realized_pnl_ = 0.0;
    double unrealized_pnl_ = 0.0;
    double total_pnl_ = 0.0;
    
    // Uncapped P&L values for analysis
    double uncapped_unrealized_pnl_ = 0.0;
    double uncapped_total_pnl_ = 0.0;
    
    double entry_value_ = 0.0;
    double current_value_ = 0.0;
    double win_count_ = 0;
    double loss_count_ = 0;
    double largest_win_ = 0.0;
    double largest_loss_ = 0.0;
    double avg_win_ = 0.0;
    double avg_loss_ = 0.0;
    double avg_trade_duration_ms_ = 0.0;
    mutable std::mutex pnl_mutex_;
    
    // USD volume tracking
    double interval_usd_volume_ = 0.0;     // USD volume for current interval
    double total_usd_volume_ = 0.0;        // Total USD volume across all intervals
    double max_interval_volume_ = 0.0; // Maximum USD volume in any interval
    double avg_interval_volume_ = 0.0; // Average USD volume per interval
    uint64_t interval_count_ = 0;          // Number of intervals tracked
    std::chrono::system_clock::time_point last_volume_reset_;  // Time of last volume reset
    mutable std::mutex volume_mutex_;      // Mutex for volume data
    
    std::vector<TradeOrder> open_orders_;
    std::vector<TradeOrder> filled_orders_;
    mutable std::mutex orders_mutex_;
    
    // Performance metrics
    std::chrono::nanoseconds avg_tick_to_trade_{0};
    std::atomic<uint64_t> trades_per_second_{0};
    std::chrono::system_clock::time_point last_metrics_update_;
    uint64_t total_trades_{0};
    uint64_t successful_trades_{0};
    mutable std::mutex metrics_mutex_;
    
    // Configuration
    std::string symbol_;
    double max_position_ = 1.0;
    double order_size_ = 0.1;
    double min_spread_ = 0.0001;
    
    // Timestamp for latency measurement
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Pre-allocated objects to avoid allocations in critical path
    TradeOrder* pending_order_ = nullptr;
    
    // Price history for volatility calculation
    static constexpr int price_history_length_ = 20;
    std::vector<double> price_history_;
    mutable std::mutex price_history_mutex_;
};

} // namespace trading 