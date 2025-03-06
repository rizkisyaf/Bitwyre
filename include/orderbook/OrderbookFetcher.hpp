#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <condition_variable>
#include <nlohmann/json.hpp>

// Define ASIO_STANDALONE before including WebSocket++
#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif

// Remove the WEBSOCKETPP_NO_TLS definition to enable TLS
// Include necessary WebSocket++ headers
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

using json = nlohmann::json;
using websocket_client = websocketpp::client<websocketpp::config::asio_tls_client>;

namespace trading {

// Pre-allocate memory for orders
constexpr size_t INITIAL_ORDERS_CAPACITY = 100;

// Use alignas for better memory alignment and cache efficiency
struct alignas(16) Order {
    double price;
    double quantity;
    
    Order(double p, double q) : price(p), quantity(q) {}
};

// Forward declaration
class Orderbook;

// Callback type for orderbook updates
using OrderbookCallback = std::function<void(const Orderbook&)>;

/**
 * @brief Represents an orderbook with bids and asks
 */
class Orderbook {
public:
    Orderbook() {
        // Pre-allocate memory to avoid reallocations
        bids_.reserve(INITIAL_ORDERS_CAPACITY);
        asks_.reserve(INITIAL_ORDERS_CAPACITY);
    }
    
    Orderbook(const Orderbook& other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        bids_ = other.bids_;
        asks_ = other.asks_;
    }
    
    Orderbook& operator=(const Orderbook& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(mutex_);
            std::lock_guard<std::mutex> lock2(other.mutex_);
            bids_ = other.bids_;
            asks_ = other.asks_;
        }
        return *this;
    }
    
    void updateBids(const std::vector<Order>& bids);
    void updateAsks(const std::vector<Order>& asks);
    
    double getBestBid() const;
    
    double getBestAsk() const;
    
    double getMidPrice() const;
    
    double getSpread() const;
    
    // Use inline for small, frequently called methods
    inline std::vector<Order> getBids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bids_;
    }
    
    inline std::vector<Order> getAsks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return asks_;
    }

private:
    std::vector<Order> bids_;
    std::vector<Order> asks_;
    mutable std::mutex mutex_;
};

/**
 * @brief Fetches orderbook data from an exchange
 */
class OrderbookFetcher {
public:
    OrderbookFetcher();
    OrderbookFetcher(const std::string& symbol);
    ~OrderbookFetcher();
    
    /**
     * @brief Connect to the exchange
     * 
     * @param exchange_url The exchange websocket URL
     * @param symbol The trading symbol
     * @return true if connection was successful
     */
    bool connect(const std::string& exchange_url, const std::string& symbol);
    
    /**
     * @brief Disconnect from the exchange
     */
    void disconnect();
    
    /**
     * @brief Register a callback for orderbook updates
     * 
     * @param callback The callback function
     */
    void registerCallback(OrderbookCallback callback);
    
    /**
     * @brief Get the latest orderbook
     * 
     * @return The latest orderbook
     */
    Orderbook getLatestOrderbook() const;
    
    /**
     * @brief Get the trading symbol
     * 
     * @return The trading symbol
     */
    std::string getSymbol() const { return symbol_; }
    
private:
    void run();
    void onMessage(websocket_client* client, websocketpp::connection_hdl hdl, 
                  websocket_client::message_ptr msg);
    void onOpen(websocket_client* client, websocketpp::connection_hdl hdl);
    void onClose(websocket_client* client, websocketpp::connection_hdl hdl);
    void onFail(websocket_client* client, websocketpp::connection_hdl hdl);
    void processOrderbookUpdate(const json& data);
    
    std::unique_ptr<websocket_client> client_;
    websocketpp::connection_hdl connection_;
    std::string symbol_;
    
    Orderbook orderbook_;
    mutable std::mutex orderbook_mutex_;
    
    std::vector<OrderbookCallback> callbacks_;
    std::mutex callbacks_mutex_;
    
    std::atomic<bool> running_{false};
    std::thread websocket_thread_;
    std::condition_variable cv_;
    
    // Pre-allocated vectors for processing updates
    std::vector<Order> bid_updates_;
    std::vector<Order> ask_updates_;
};

} // namespace trading 