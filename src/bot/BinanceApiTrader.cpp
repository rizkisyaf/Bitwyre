#include "bot/BinanceApiTrader.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace trading {

BinanceApiTrader::BinanceApiTrader(const std::string& api_key, const std::string& secret_key, TradingMode mode)
    : api_key_(api_key), secret_key_(secret_key), trading_mode_(mode), initialized_(false) {
    // Set the base URL based on trading mode
    if (trading_mode_ == TradingMode::USDM_FUTURES) {
        base_url_ = "https://fapi.binance.com";
        std::cout << "Using USDM Futures trading mode with base URL: " << base_url_ << std::endl;
    } else {
        base_url_ = "https://api.binance.com";
        std::cout << "Using Spot trading mode with base URL: " << base_url_ << std::endl;
    }
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl_ = curl_easy_init();
    
    if (!curl_) {
        std::cerr << "ERROR: Failed to initialize curl" << std::endl;
    } else {
        std::cout << "CURL initialized successfully" << std::endl;
    }
    
    // Mask API key for logging
    std::string masked_key = api_key_.substr(0, 4) + "..." + api_key_.substr(api_key_.length() - 4);
    std::cout << "API Key configured: " << masked_key << std::endl;
}

BinanceApiTrader::~BinanceApiTrader() {
    // Clean up curl
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
    curl_global_cleanup();
}

bool BinanceApiTrader::initialize() {
    // Test the API connection by getting account info
    try {
        std::cout << "Initializing Binance API connection..." << std::endl;
        std::cout << "Testing API connection with account info request..." << std::endl;
        
        auto account_info = getAccountInfo();
        
        // Log the response for debugging
        std::cout << "Successfully connected to Binance API" << std::endl;
        
        // For futures trading, check permissions
        if (trading_mode_ == TradingMode::USDM_FUTURES) {
            std::cout << "Checking futures trading permissions..." << std::endl;
            
            // Check if we have futures trading enabled
            if (account_info.contains("canTrade")) {
                if (account_info["canTrade"].get<bool>()) {
                    std::cout << "Futures trading is enabled for this account" << std::endl;
                } else {
                    std::cerr << "ERROR: Futures trading is NOT enabled for this account" << std::endl;
                    return false;
                }
            } else {
                std::cout << "WARNING: Could not verify futures trading permission" << std::endl;
            }
        }
        
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to connect to Binance API: " << e.what() << std::endl;
        return false;
    }
}

// Helper method to get the base URL
std::string BinanceApiTrader::getBaseUrl() const {
    return base_url_;
}

// Helper method to round price to the correct tick size
double BinanceApiTrader::roundToTickSize(double price, const std::string& symbol) {
    // Default tick size and precision
    double tick_size = 0.01;
    int decimal_precision = 2;
    
    // Set specific tick sizes for different symbols
    std::string symbol_lower = symbol;
    std::transform(symbol_lower.begin(), symbol_lower.end(), symbol_lower.begin(), ::tolower);
    
    if (symbol_lower == "btcusdt") {
        // For BTCUSDT futures, the tick size is 0.10 according to Binance API
        tick_size = 0.10;
        decimal_precision = 1;
        
        std::cout << "BTCUSDT original price: " << price << std::endl;
    } else if (symbol_lower == "ethusdt") {
        tick_size = 0.01;
        decimal_precision = 2;
    } else if (symbol_lower == "bnbusdt") {
        tick_size = 0.01;
        decimal_precision = 2;
    }
    
    // For BTCUSDT, ensure we're using the exact tick size of 0.10
    if (symbol_lower == "btcusdt") {
        // First, truncate to 1 decimal place to avoid floating point issues
        double truncated = std::floor(price * 10) / 10.0;
        
        // Then round to the nearest tick size
        double rounded_price = std::round(truncated / tick_size) * tick_size;
        
        // Format with appropriate decimal precision
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(decimal_precision) << rounded_price;
        
        // Convert back to double
        double result = std::stod(ss.str());
        
        std::cout << "Price rounding for BTCUSDT:"
                  << "\n- Original: " << price
                  << "\n- Truncated: " << truncated
                  << "\n- Rounded: " << rounded_price
                  << "\n- Final: " << result
                  << "\n- Tick size: " << tick_size << std::endl;
        
        return result;
    } else {
        // For other symbols, use the standard rounding
        double rounded_price = std::round(price / tick_size) * tick_size;
        
        // Format with appropriate decimal precision to avoid floating point issues
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(decimal_precision) << rounded_price;
        std::string price_str = ss.str();
        
        // Convert back to double
        return std::stod(price_str);
    }
}

bool BinanceApiTrader::placeOrder(const TradeOrder& order, const std::string& symbol) {
    if (!initialized_) {
        std::cerr << "❌ API trader not initialized" << std::endl;
        return false;
    }
    
    try {
        std::string side = (order.type == TradeOrder::Type::BUY) ? "BUY" : "SELL";
        
        // Format price and quantity with appropriate precision
        double price = order.price;
        double quantity = order.quantity;
        
        // Adjust quantity and price based on symbol
        double adjustedPrice = price;
        double adjustedQuantity = quantity;
        std::string priceStr;
        std::string quantityStr;
        
        if (symbol == "BTCUSDT" && trading_mode_ == TradingMode::USDM_FUTURES) {
            // For BTCUSDT, the tick size is exactly 0.1
            const double TICK_SIZE = 0.1;
            const double MIN_NOTIONAL = 100.0;
            
            // Round to the nearest valid tick size
            if (side == "BUY") {
                // Round up for buy orders to ensure we're above the market price
                adjustedPrice = std::ceil(price * 10.0) / 10.0;
            } else {
                // Round down for sell orders to ensure we're below the market price
                adjustedPrice = std::floor(price * 10.0) / 10.0;
            }
            
            std::ostringstream price_ss;
            price_ss << std::fixed << std::setprecision(1) << adjustedPrice;
            priceStr = price_ss.str();
            
            // Calculate minimum quantity needed for 100 USDT notional value
            adjustedQuantity = std::ceil((MIN_NOTIONAL / adjustedPrice) * 100.0) / 100.0;  // Round to 2 decimal places
            
            std::ostringstream quantity_ss;
            quantity_ss << std::fixed << std::setprecision(2) << adjustedQuantity;  // Use 2 decimal places for BTC quantity
            quantityStr = quantity_ss.str();
            
            double notionalValue = adjustedPrice * adjustedQuantity;
            std::cout << "📊 Order Details [" << symbol << "]:"
                      << "\n  Type: " << side
                      << "\n  Price: " << priceStr << " USDT"
                      << "\n  Quantity: " << quantityStr << " BTC"
                      << "\n  Value: " << notionalValue << " USDT" << std::endl;
            
            double priceMod = std::fmod(adjustedPrice, TICK_SIZE);
            if (std::abs(priceMod) > 0.00001) {
                std::cerr << "❌ Price " << adjustedPrice << " does not align with tick size " << TICK_SIZE << std::endl;
                return false;
            }
        } else if (symbol == "ETHUSDT" && trading_mode_ == TradingMode::USDM_FUTURES) {
            // For ETHUSDT, ensure price is a multiple of 0.01 (tick size)
            const double TICK_SIZE = 0.01;
            adjustedPrice = std::floor(price / TICK_SIZE) * TICK_SIZE;
            
            // Calculate notional value (price * quantity)
            double notionalValue = adjustedPrice * adjustedQuantity;
            
            // Ensure minimum notional value of 100 USDT
            if (notionalValue < 100.0) {
                adjustedQuantity = 100.0 / adjustedPrice;
                // Round up to 3 decimal places for ETH
                adjustedQuantity = std::ceil(adjustedQuantity * 1000) / 1000;
                
                std::cout << "Adjusting quantity from " << quantity << " to " << adjustedQuantity 
                          << " to meet minimum notional value of 100 USDT" << std::endl;
                
                notionalValue = adjustedPrice * adjustedQuantity;
                std::cout << "New notional value: " << notionalValue << " USDT" << std::endl;
            }
            
            // Format with 2 decimal places for price (0.01 tick size)
            std::ostringstream price_ss;
            price_ss << std::fixed << std::setprecision(2) << adjustedPrice;
            priceStr = price_ss.str();
            
            // Format with 3 decimal places for quantity
            std::ostringstream qty_ss;
            qty_ss << std::fixed << std::setprecision(3) << adjustedQuantity;
            quantityStr = qty_ss.str();
            
            std::cout << "Adjusted price for ETHUSDT: " << priceStr << " (tick size: 0.01)" << std::endl;
        } else if (symbol == "BNBUSDT" && trading_mode_ == TradingMode::USDM_FUTURES) {
            // For BNBUSDT, ensure price is a multiple of 0.01 (tick size)
            const double TICK_SIZE = 0.01;
            adjustedPrice = std::floor(price / TICK_SIZE) * TICK_SIZE;
            
            // Calculate notional value (price * quantity)
            double notionalValue = adjustedPrice * adjustedQuantity;
            
            // Ensure minimum notional value of 100 USDT
            if (notionalValue < 100.0) {
                adjustedQuantity = 100.0 / adjustedPrice;
                // Round up to 2 decimal places for BNB
                adjustedQuantity = std::ceil(adjustedQuantity * 100) / 100;
                
                std::cout << "Adjusting quantity from " << quantity << " to " << adjustedQuantity 
                          << " to meet minimum notional value of 100 USDT" << std::endl;
                
                notionalValue = adjustedPrice * adjustedQuantity;
                std::cout << "New notional value: " << notionalValue << " USDT" << std::endl;
            }
            
            // Format with 2 decimal places for price (0.01 tick size)
            std::ostringstream price_ss;
            price_ss << std::fixed << std::setprecision(2) << adjustedPrice;
            priceStr = price_ss.str();
            
            // Format with 2 decimal places for quantity
            std::ostringstream qty_ss;
            qty_ss << std::fixed << std::setprecision(2) << adjustedQuantity;
            quantityStr = qty_ss.str();
            
            std::cout << "Adjusted price for BNBUSDT: " << priceStr << " (tick size: 0.01)" << std::endl;
        } else {
            // Default formatting for other symbols
            // Format with 2 decimal places for price
            std::ostringstream price_ss;
            price_ss << std::fixed << std::setprecision(2) << price;
            priceStr = price_ss.str();
            
            // Format with 8 decimal places for quantity
            std::ostringstream qty_ss;
            qty_ss << std::fixed << std::setprecision(8) << quantity;
            quantityStr = qty_ss.str();
        }
        
        std::cout << "Placing " << side << " order for " << symbol << ": " << adjustedQuantity << " @ " << adjustedPrice << std::endl;
        
        // Construct the order parameters
        std::string endpoint = "/fapi/v1/order";
        if (trading_mode_ == TradingMode::SPOT) {
            endpoint = "/api/v3/order";
        }
        
        // Build the query string
        std::string query_string = "symbol=" + symbol + 
                                  "&side=" + side + 
                                  "&type=LIMIT" + 
                                  "&timeInForce=GTC" + 
                                  "&quantity=" + quantityStr + 
                                  "&price=" + priceStr + 
                                  "&reduceOnly=false" +
                                  "&timestamp=" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count());
        
        // Send the signed request
        auto response = sendSignedRequest("POST", endpoint, query_string);
        
        // Check if the order was placed successfully
        if (response.contains("orderId")) {
            std::string order_id;
            if (response["orderId"].is_string()) {
                order_id = response["orderId"].get<std::string>();
            } else if (response["orderId"].is_number()) {
                order_id = std::to_string(response["orderId"].get<int64_t>());
            }
            std::cout << "✅ Order placed successfully [ID: " << order_id << "]" << std::endl;
            return true;
        } else {
            std::cerr << "❌ Failed to place order" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION placing order: " << e.what() << std::endl;
        return false;
    }
}

bool BinanceApiTrader::cancelOrder(const std::string& order_id, const std::string& symbol) {
    try {
        // Build query string
        std::stringstream ss;
        ss << "symbol=" << symbol
           << "&orderId=" << order_id
           << "&timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request to the appropriate API
        std::string endpoint = (trading_mode_ == TradingMode::USDM_FUTURES) ? 
                              "/fapi/v1/order" : "/api/v3/order";
        
        auto response = sendSignedRequest("DELETE", endpoint, ss.str());
        
        // Check for errors
        if (response.contains("code") && response["code"].is_number()) {
            std::cerr << "Error canceling order: " << response["msg"].get<std::string>() << std::endl;
            return false;
        }
        
        std::cout << "Order canceled successfully: " << order_id << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception canceling order: " << e.what() << std::endl;
        return false;
    }
}

TradeOrder BinanceApiTrader::getOrderStatus(const std::string& order_id, const std::string& symbol) {
    try {
        // Build query string
        std::stringstream ss;
        ss << "symbol=" << symbol
           << "&orderId=" << order_id
           << "&timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request to the appropriate API
        std::string endpoint = (trading_mode_ == TradingMode::USDM_FUTURES) ? 
                              "/fapi/v1/order" : "/api/v3/order";
        
        auto response = sendSignedRequest("GET", endpoint, ss.str());
        
        // Convert response to TradeOrder
        return convertBinanceOrderToTradeOrder(response);
    } catch (const std::exception& e) {
        std::cerr << "Exception getting order status: " << e.what() << std::endl;
        // Return an empty order with REJECTED status
        TradeOrder empty_order(TradeOrder::Type::BUY, 0.0, 0.0);
        empty_order.status = TradeOrder::Status::REJECTED;
        return empty_order;
    }
}

nlohmann::json BinanceApiTrader::getAccountInfo() {
    try {
        // Build query string
        std::stringstream ss;
        ss << "timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request to the appropriate API
        std::string endpoint = (trading_mode_ == TradingMode::USDM_FUTURES) ? 
                              "/fapi/v2/account" : "/api/v3/account";
        
        return sendSignedRequest("GET", endpoint, ss.str());
    } catch (const std::exception& e) {
        std::cerr << "Exception getting account info: " << e.what() << std::endl;
        throw;
    }
}

std::vector<TradeOrder> BinanceApiTrader::getOpenOrders(const std::string& symbol) {
    try {
        // Build query string
        std::stringstream ss;
        ss << "timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        if (!symbol.empty()) {
            ss << "&symbol=" << symbol;
        }
        
        // Send the request to the appropriate API
        std::string endpoint = (trading_mode_ == TradingMode::USDM_FUTURES) ? 
                              "/fapi/v1/openOrders" : "/api/v3/openOrders";
        
        auto response = sendSignedRequest("GET", endpoint, ss.str());
        
        // Convert response to vector of TradeOrder
        std::vector<TradeOrder> orders;
        for (const auto& order_json : response) {
            orders.push_back(convertBinanceOrderToTradeOrder(order_json));
        }
        
        return orders;
    } catch (const std::exception& e) {
        std::cerr << "Exception getting open orders: " << e.what() << std::endl;
        return {};
    }
}

bool BinanceApiTrader::setLeverage(const std::string& symbol, int leverage) {
    if (trading_mode_ != TradingMode::USDM_FUTURES) {
        std::cerr << "Leverage can only be set in USDM futures mode" << std::endl;
        return false;
    }
    
    try {
        // Build query string
        std::stringstream ss;
        ss << "symbol=" << symbol
           << "&leverage=" << leverage
           << "&timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request
        auto response = sendSignedRequest("POST", "/fapi/v1/leverage", ss.str());
        
        // Check for errors
        if (response.contains("code") && response["code"].is_number()) {
            std::cerr << "Error setting leverage: " << response["msg"].get<std::string>() << std::endl;
            return false;
        }
        
        std::cout << "Leverage set to " << leverage << "x for " << symbol << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception setting leverage: " << e.what() << std::endl;
        return false;
    }
}

bool BinanceApiTrader::setMarginType(const std::string& symbol, bool isolated) {
    if (trading_mode_ != TradingMode::USDM_FUTURES) {
        std::cerr << "Margin type can only be set in USDM futures mode" << std::endl;
        return false;
    }
    
    try {
        // Build query string
        std::stringstream ss;
        ss << "symbol=" << symbol
           << "&marginType=" << (isolated ? "ISOLATED" : "CROSSED")
           << "&timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request
        auto response = sendSignedRequest("POST", "/fapi/v1/marginType", ss.str());
        
        // Check for errors
        if (response.contains("code") && response["code"].is_number()) {
            // Code 200 means the margin type is already set to the requested type
            if (response["code"].get<int>() == 200) {
                std::cout << "Margin type already set to " << (isolated ? "ISOLATED" : "CROSSED") << " for " << symbol << std::endl;
                return true;
            }
            
            std::cerr << "Error setting margin type: " << response["msg"].get<std::string>() << std::endl;
            return false;
        }
        
        std::cout << "Margin type set to " << (isolated ? "ISOLATED" : "CROSSED") << " for " << symbol << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception setting margin type: " << e.what() << std::endl;
        return false;
    }
}

nlohmann::json BinanceApiTrader::sendSignedRequest(const std::string& method, const std::string& endpoint, 
                                                 const std::string& query_string) {
    // Create signature
    std::string signature = createSignature(query_string);
    
    // Append signature to query string
    std::string full_query = query_string + "&signature=" + signature;
    
    std::cout << "Sending signed request: " << method << " " << endpoint << std::endl;
    
    // Send the request
    return sendRequest(method, endpoint, full_query);
}

nlohmann::json BinanceApiTrader::sendRequest(const std::string& method, const std::string& endpoint, 
                                           const std::string& query_string) {
    std::lock_guard<std::mutex> lock(curl_mutex_);
    
    if (!curl_) {
        throw std::runtime_error("Curl not initialized");
    }
    
    // Reset curl
    curl_easy_reset(curl_);
    
    // Build URL
    std::string url = base_url_ + endpoint;
    if (!query_string.empty() && method == "GET") {
        url += "?" + query_string;
    }
    
    std::cout << "API Request: " << method << " " << url << std::endl;
    
    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    
    // Set method
    if (method == "POST") {
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        if (!query_string.empty()) {
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, query_string.c_str());
        }
    } else if (method == "DELETE") {
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
        if (!query_string.empty()) {
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, query_string.c_str());
        }
    }
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("X-MBX-APIKEY: " + api_key_).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    
    // Set write callback
    std::string response_string;
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_string);
    
    // Set verbose mode for debugging
    curl_easy_setopt(curl_, CURLOPT_VERBOSE, 0L);  // Disable verbose mode
    
    // Set timeout
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 10L);
    
    // Perform the request
    CURLcode res = curl_easy_perform(curl_);
    
    // Clean up headers
    curl_slist_free_all(headers);
    
    // Check for errors
    if (res != CURLE_OK) {
        std::string error_msg = std::string("Curl request failed: ") + curl_easy_strerror(res);
        std::cerr << "❌ API Error: " << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }
    
    // Get HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
    
    // Parse response
    try {
        // Only log response for non-200 status codes
        if (http_code >= 400) {
            std::cout << "❌ API Response: " << response_string.substr(0, 500) << 
                        (response_string.length() > 500 ? "..." : "") << std::endl;
        }
        
        // Check for HTTP error codes
        if (http_code >= 400) {
            nlohmann::json error_json = nlohmann::json::parse(response_string);
            std::string error_msg;
            
            if (error_json.contains("code") && error_json.contains("msg")) {
                error_msg = "API Error: " + error_json["msg"].get<std::string>() + 
                            " (code: " + std::to_string(error_json["code"].get<int>()) + ")";
            } else {
                error_msg = "HTTP Error: " + std::to_string(http_code);
            }
            
            std::cerr << "❌ " << error_msg << std::endl;
            throw std::runtime_error(error_msg);
        }
        
        return nlohmann::json::parse(response_string);
    } catch (const nlohmann::json::parse_error& e) {
        std::string error_msg = std::string("Failed to parse response: ") + e.what() + "\nResponse: " + response_string;
        std::cerr << "ERROR: " << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }
}

std::string BinanceApiTrader::createSignature(const std::string& query_string) {
    // Create HMAC-SHA256 signature
    unsigned char* digest = HMAC(EVP_sha256(), secret_key_.c_str(), secret_key_.length(),
                               (unsigned char*)query_string.c_str(), query_string.length(),
                               nullptr, nullptr);
    
    // Convert to hex string
    std::stringstream ss;
    for (int i = 0; i < 32; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    
    return ss.str();
}

size_t BinanceApiTrader::curlWriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t new_length = size * nmemb;
    try {
        s->append((char*)contents, new_length);
        return new_length;
    } catch (const std::bad_alloc& e) {
        // Handle memory problem
        return 0;
    }
}

TradeOrder BinanceApiTrader::convertBinanceOrderToTradeOrder(const nlohmann::json& binance_order) {
    // Extract order details
    std::string order_id;
    if (binance_order["orderId"].is_string()) {
        order_id = binance_order["orderId"].get<std::string>();
    } else if (binance_order["orderId"].is_number()) {
        order_id = std::to_string(binance_order["orderId"].get<int64_t>());
    }
    
    std::string side = binance_order["side"].get<std::string>();
    double price = std::stod(binance_order["price"].get<std::string>());
    double quantity = std::stod(binance_order["origQty"].get<std::string>());
    double filled_quantity = std::stod(binance_order["executedQty"].get<std::string>());
    std::string status_str = binance_order["status"].get<std::string>();
    
    // Convert side to TradeOrder::Type
    TradeOrder::Type type = (side == "BUY") ? TradeOrder::Type::BUY : TradeOrder::Type::SELL;
    
    // Create TradeOrder
    TradeOrder order(type, price, quantity);
    order.id = order_id;
    order.filled_quantity = filled_quantity;
    
    // Convert status
    if (status_str == "NEW" || status_str == "PARTIALLY_FILLED") {
        order.status = TradeOrder::Status::PENDING;
    } else if (status_str == "FILLED") {
        order.status = TradeOrder::Status::FILLED;
    } else if (status_str == "CANCELED" || status_str == "EXPIRED" || status_str == "REJECTED") {
        order.status = TradeOrder::Status::CANCELED;
    } else {
        order.status = TradeOrder::Status::REJECTED;
    }
    
    return order;
}

std::string BinanceApiTrader::convertTradeOrderTypeToBinanceOrderSide(TradeOrder::Type type) {
    return (type == TradeOrder::Type::BUY) ? "BUY" : "SELL";
}

bool BinanceApiTrader::cancelAllOrders(const std::string& symbol) {
    try {
        // Build query string
        std::stringstream ss;
        ss << "symbol=" << symbol
           << "&timestamp=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Send the request to the appropriate API
        std::string endpoint = (trading_mode_ == TradingMode::USDM_FUTURES) ? 
                              "/fapi/v1/allOpenOrders" : "/api/v3/openOrders";
        
        auto response = sendSignedRequest("DELETE", endpoint, ss.str());
        
        // Check for errors
        if (response.contains("code") && response["code"].is_number() && response["code"].get<int>() != 200) {
            std::cerr << "Error canceling all orders: " << response["msg"].get<std::string>() << std::endl;
            return false;
        }
        
        std::cout << "All orders for " << symbol << " canceled successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception canceling all orders: " << e.what() << std::endl;
        return false;
    }
}

} // namespace trading 