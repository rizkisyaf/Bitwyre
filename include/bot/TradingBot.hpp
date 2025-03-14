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

// Forward declaration
class BinanceApiTrader;

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
    // Add a new constructor that takes API keys for real trading
    TradingBot(std::shared_ptr<OrderbookFetcher> fetcher, 
               std::shared_ptr<TradingModel> model,
               double initial_balance,
               const std::string& api_key,
               const std::string& secret_key);
               
    ~TradingBot();
    
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
     * @brief Get P&L metrics
     * 
     * @return P&L metrics as JSON
     */
    nlohmann::json getPnLMetrics() const;
    
    // Risk management settings
    void setStopLossPercentage(double percentage);
    void setMaxDrawdownPercentage(double percentage);
    
    /**
     * @brief Emergency stop - stops trading and cancels all open orders
     * @param timeout_ms Maximum time to wait for graceful shutdown in milliseconds
     * @return true if shutdown was graceful, false if forced
     */
    bool emergencyStop(int timeout_ms = 5000);
    
    /**
     * @brief Cancel all open orders
     * 
     * @return true if all orders were canceled successfully
     */
    bool cancelAllOrders();
    
    // Save trading history to a file for model retraining
    void saveTradingHistory(const std::string& filename = "trading_history.csv");
    
private:
    void run();
    void onOrderbookUpdate(const Orderbook& orderbook);
    void processModelPrediction(double prediction, const Orderbook& orderbook_snapshot, const std::vector<float>& features);
    bool placeOrder(const TradeOrder& order);
    bool cancelOrder(const std::string& order_id);
    void updateOrderStatus();
    
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
    
    // Thread management
    std::atomic<bool> running_{false};
    std::atomic<bool> emergency_stop_{false};
    std::atomic<bool> force_stop_{false};
    std::thread trading_thread_;
    std::thread websocket_thread_;
    std::condition_variable cv_;
    mutable std::mutex mutex_;
    
    // Shutdown management
    void shutdownThreads(bool force = false);
    bool waitForThreads(int timeout_ms);
    void closeWebSocket();
    
    // Timeouts
    static constexpr int ORDER_CANCEL_TIMEOUT_MS = 2000;  // 2 seconds
    static constexpr int WEBSOCKET_TIMEOUT_MS = 1000;     // 1 second
    static constexpr int THREAD_JOIN_TIMEOUT_MS = 2000;   // 2 seconds
    
    // Thread-safe state tracking
    std::atomic<int> active_operations_{0};
    std::atomic<bool> websocket_connected_{false};
    
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
    
    // Configuration
    std::string symbol_;
    std::string exchange_url_ = "wss://stream.binance.com:9443/ws";  // Default WebSocket URL
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
    
    // API trader for real trading
    std::shared_ptr<BinanceApiTrader> api_trader_ = nullptr;
    
    // Trading parameters
    int leverage_ = 5;  // Default 5x leverage
    
    // Structure to track trading decisions and outcomes
    struct TradingDecision {
        std::chrono::system_clock::time_point timestamp;
        std::vector<float> features;
        double prediction;
        double signal_strength;
        TradeOrder::Type action;  // BUY, SELL, or NONE
        double entry_price;
        double exit_price;
        double profit_loss;
        bool completed;
    };
    
    std::vector<TradingDecision> trading_history_;
    std::mutex history_mutex_;
    
    // Method to record a new trading decision
    void recordTradingDecision(const std::vector<float>& features, double prediction, 
                              double signal_strength, TradeOrder::Type action, double price);
    
    // Method to update a trading decision with its outcome
    void updateTradingDecision(const std::string& order_id, double exit_price, double profit_loss);
};

} // namespace trading 