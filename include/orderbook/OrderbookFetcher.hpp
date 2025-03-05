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

/**
 * @brief Represents an order in the orderbook
 */
struct Order {
    double price;
    double quantity;
    
    Order(double p, double q) : price(p), quantity(q) {}
};

/**
 * @brief Represents a snapshot of the orderbook at a given time
 */
class Orderbook {
public:
    Orderbook() = default;
    
    // Copy constructor
    Orderbook(const Orderbook& other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        bids_ = other.bids_;
        asks_ = other.asks_;
    }
    
    // Assignment operator
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
    
    // Get the best bid price
    double getBestBid() const;
    // Get the best ask price
    double getBestAsk() const;
    // Get the mid price
    double getMidPrice() const;
    // Get the spread
    double getSpread() const;
    
    // Get all bids
    std::vector<Order> getBids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bids_;
    }
    
    // Get all asks
    std::vector<Order> getAsks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return asks_;
    }
    
private:
    std::vector<Order> bids_;
    std::vector<Order> asks_;
    mutable std::mutex mutex_;
};

/**
 * @brief Fetches orderbook data from an exchange using websockets
 */
class OrderbookFetcher {
public:
    using OrderbookCallback = std::function<void(const Orderbook&)>;
    
    OrderbookFetcher();
    ~OrderbookFetcher();
    
    /**
     * @brief Connect to the exchange websocket
     * @param exchange_url The URL of the exchange websocket
     * @param symbol The trading pair symbol
     * @return true if connection was successful, false otherwise
     */
    bool connect(const std::string& exchange_url, const std::string& symbol);
    
    /**
     * @brief Disconnect from the exchange websocket
     */
    void disconnect();
    
    /**
     * @brief Register a callback to be called when the orderbook is updated
     * @param callback The callback function
     */
    void registerCallback(OrderbookCallback callback);
    
    /**
     * @brief Get the latest orderbook snapshot
     * @return The latest orderbook snapshot
     */
    Orderbook getLatestOrderbook() const;
    
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
    
    std::atomic<bool> running_;
    std::thread websocket_thread_;
    std::condition_variable cv_;
};

} // namespace trading 