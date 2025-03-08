#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <openssl/hmac.h>
#include "bot/TradingBot.hpp"

namespace trading {

/**
 * @brief Handles real trading with Binance API
 */
class BinanceApiTrader {
public:
    enum class TradingMode {
        SPOT,
        USDM_FUTURES
    };
    
    BinanceApiTrader(const std::string& api_key, const std::string& secret_key, TradingMode mode = TradingMode::USDM_FUTURES);
    ~BinanceApiTrader();
    
    /**
     * @brief Initialize the API trader
     * 
     * @return true if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Get the current trading mode
     * 
     * @return The trading mode (SPOT or USDM_FUTURES)
     */
    TradingMode getTradingMode() const { return trading_mode_; }
    
    /**
     * @brief Get the base URL for API requests
     * 
     * @return The base URL string
     */
    std::string getBaseUrl() const;
    
    /**
     * @brief Place a real order on Binance
     * 
     * @param order The order to place
     * @return true if the order was placed successfully
     */
    bool placeOrder(const TradeOrder& order, const std::string& symbol);
    
    /**
     * @brief Cancel an order on Binance
     * 
     * @param order_id The ID of the order to cancel
     * @param symbol The trading symbol
     * @return true if the order was canceled successfully
     */
    bool cancelOrder(const std::string& order_id, const std::string& symbol);
    
    /**
     * @brief Cancel all open orders for a symbol
     * 
     * @param symbol The trading symbol
     * @return true if all orders were canceled successfully
     */
    bool cancelAllOrders(const std::string& symbol);
    
    /**
     * @brief Get the status of an order
     * 
     * @param order_id The ID of the order
     * @param symbol The trading symbol
     * @return The updated order
     */
    TradeOrder getOrderStatus(const std::string& order_id, const std::string& symbol);
    
    /**
     * @brief Get account information
     * 
     * @return Account information as JSON
     */
    nlohmann::json getAccountInfo();
    
    /**
     * @brief Get open orders
     * 
     * @param symbol The trading symbol (optional)
     * @return Open orders as a vector
     */
    std::vector<TradeOrder> getOpenOrders(const std::string& symbol = "");
    
    /**
     * @brief Set leverage for futures trading
     * 
     * @param symbol The trading symbol
     * @param leverage The leverage value (1-125)
     * @return true if successful
     */
    bool setLeverage(const std::string& symbol, int leverage);
    
    /**
     * @brief Set margin type for futures trading
     * 
     * @param symbol The trading symbol
     * @param isolated Whether to use isolated margin (true) or cross margin (false)
     * @return true if successful
     */
    bool setMarginType(const std::string& symbol, bool isolated);
    
private:
    // Helper methods for API requests
    nlohmann::json sendSignedRequest(const std::string& method, const std::string& endpoint, 
                                    const std::string& query_string = "");
    nlohmann::json sendRequest(const std::string& method, const std::string& endpoint, 
                              const std::string& query_string = "");
    std::string createSignature(const std::string& query_string);
    static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, std::string* s);
    
    // Convert between TradeOrder and Binance API formats
    TradeOrder convertBinanceOrderToTradeOrder(const nlohmann::json& binance_order);
    std::string convertTradeOrderTypeToBinanceOrderSide(TradeOrder::Type type);
    
    // Helper method to round price to the correct tick size
    double roundToTickSize(double price, const std::string& symbol);
    
    std::string api_key_;
    std::string secret_key_;
    std::string base_url_;
    TradingMode trading_mode_;
    bool initialized_ = false;
    CURL* curl_ = nullptr;
    std::mutex curl_mutex_;
};

} // namespace trading 