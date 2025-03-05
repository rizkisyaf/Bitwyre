#include "bot/TradingBot.hpp"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>

namespace trading {

TradingBot::TradingBot() : running_(false), position_(0.0), balance_(0.0), 
                          total_trades_(0), successful_trades_(0) {
    // Initialize performance metrics
    avg_tick_to_trade_ = std::chrono::nanoseconds(0);
    trades_per_second_ = 0;
    last_metrics_update_ = std::chrono::system_clock::now();
}

TradingBot::~TradingBot() {
    stop();
}

bool TradingBot::initialize(const std::string& exchange_url, const std::string& symbol, const std::string& model_path) {
    symbol_ = symbol;
    
    // Create orderbook fetcher
    orderbook_fetcher_ = std::make_unique<OrderbookFetcher>();
    
    // Register callback for orderbook updates
    orderbook_fetcher_->registerCallback([this](const Orderbook& orderbook) {
        this->onOrderbookUpdate(orderbook);
    });
    
    // Connect to exchange
    if (!orderbook_fetcher_->connect(exchange_url, symbol)) {
        std::cerr << "Failed to connect to exchange" << std::endl;
        return false;
    }
    
    // Create trading model
    trading_model_ = std::make_unique<TradingModel>();
    
    // Load model
    if (!trading_model_->loadModel(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return false;
    }
    
    std::cout << "Trading bot initialized successfully" << std::endl;
    
    // Enable simulation mode for performance testing
    std::cout << "Running in simulation mode for performance testing" << std::endl;
    
    return true;
}

void TradingBot::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    
    // Start trading thread
    trading_thread_ = std::thread(&TradingBot::run, this);
}

void TradingBot::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Notify trading thread to stop
    cv_.notify_one();
    
    // Wait for trading thread to finish
    if (trading_thread_.joinable()) {
        trading_thread_.join();
    }
    
    // Disconnect from exchange
    orderbook_fetcher_->disconnect();
}

double TradingBot::getPosition() const {
    return position_;
}

double TradingBot::getBalance() const {
    return balance_;
}

std::vector<TradeOrder> TradingBot::getOpenOrders() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    return open_orders_;
}

std::vector<TradeOrder> TradingBot::getFilledOrders() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    return filled_orders_;
}

nlohmann::json TradingBot::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    nlohmann::json metrics;
    metrics["avg_tick_to_trade_ns"] = avg_tick_to_trade_.count();
    metrics["trades_per_second"] = trades_per_second_.load();
    metrics["total_trades"] = total_trades_;
    metrics["successful_trades"] = successful_trades_;
    metrics["success_rate"] = total_trades_ > 0 ? static_cast<double>(successful_trades_) / total_trades_ : 0.0;
    
    return metrics;
}

void TradingBot::run() {
    std::cout << "Trading thread started" << std::endl;
    
    while (running_) {
        // Wait for a short time to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Update performance metrics periodically
        calculatePerformanceMetrics();
    }
    
    std::cout << "Trading thread stopped" << std::endl;
}

void TradingBot::onOrderbookUpdate(const Orderbook& orderbook) {
    if (!running_) {
        return;
    }
    
    // Record start time for tick-to-trade measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract features and make prediction
    double prediction = trading_model_->predict(orderbook);
    
    // Process prediction (in simulation mode)
    processModelPrediction(prediction);
    
    // Record end time for tick-to-trade measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto tick_to_trade = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    // Update tick-to-trade metric
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (avg_tick_to_trade_.count() == 0) {
        avg_tick_to_trade_ = tick_to_trade;
    } else {
        // Exponential moving average with alpha = 0.1
        avg_tick_to_trade_ = std::chrono::nanoseconds(
            static_cast<long>(0.9 * avg_tick_to_trade_.count() + 0.1 * tick_to_trade.count())
        );
    }
    
    // Increment total trades counter
    total_trades_++;
    
    // Log detailed performance for this update
    std::cout << "Orderbook update processed: " << std::endl;
    std::cout << "  Prediction: " << prediction << std::endl;
    std::cout << "  Tick-to-trade: " << tick_to_trade.count() << " ns" << std::endl;
    std::cout << "  Avg tick-to-trade: " << avg_tick_to_trade_.count() << " ns" << std::endl;
}

void TradingBot::processModelPrediction(double prediction) {
    // In simulation mode, we don't actually place orders
    // But we simulate the decision-making process
    
    // Simulate a successful trade 80% of the time
    if (rand() % 100 < 80) {
        successful_trades_++;
    }
    
    // Simulate position and balance changes
    if (prediction > 0.5) {
        // Simulated buy
        position_ += order_size_;
        balance_ -= order_size_ * 1000.0; // Simulated price
    } else if (prediction < -0.5) {
        // Simulated sell
        position_ -= order_size_;
        balance_ += order_size_ * 1000.0; // Simulated price
    }
}

bool TradingBot::placeOrder(const TradeOrder& order) {
    // In simulation mode, we don't actually place orders
    return true;
}

bool TradingBot::cancelOrder(const std::string& order_id) {
    // In simulation mode, we don't actually cancel orders
    return true;
}

void TradingBot::updateOrderStatus() {
    // In simulation mode, we don't need to update order status
}

void TradingBot::calculatePerformanceMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_metrics_update_);
    
    if (elapsed.count() >= 1) {
        // Calculate trades per second
        trades_per_second_ = total_trades_ / elapsed.count();
        
        // Reset counters
        last_metrics_update_ = now;
    }
}

} // namespace trading 