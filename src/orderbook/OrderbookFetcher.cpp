#include "orderbook/OrderbookFetcher.hpp"

#include <algorithm>
#include <iostream>
#include <chrono>

namespace trading {

// Orderbook implementation
void Orderbook::updateBids(const std::vector<Order>& bids) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update bids
    for (const auto& bid : bids) {
        // If quantity is 0, remove the price level
        if (bid.quantity == 0.0) {
            bids_.erase(
                std::remove_if(bids_.begin(), bids_.end(),
                    [&bid](const Order& o) { return o.price == bid.price; }),
                bids_.end()
            );
        } else {
            // Check if price level exists
            auto it = std::find_if(bids_.begin(), bids_.end(),
                [&bid](const Order& o) { return o.price == bid.price; });
            
            if (it != bids_.end()) {
                // Update quantity
                it->quantity = bid.quantity;
            } else {
                // Add new price level
                bids_.push_back(bid);
            }
        }
    }
    
    // Sort bids in descending order by price
    std::sort(bids_.begin(), bids_.end(),
        [](const Order& a, const Order& b) { return a.price > b.price; });
}

void Orderbook::updateAsks(const std::vector<Order>& asks) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update asks
    for (const auto& ask : asks) {
        // If quantity is 0, remove the price level
        if (ask.quantity == 0.0) {
            asks_.erase(
                std::remove_if(asks_.begin(), asks_.end(),
                    [&ask](const Order& o) { return o.price == ask.price; }),
                asks_.end()
            );
        } else {
            // Check if price level exists
            auto it = std::find_if(asks_.begin(), asks_.end(),
                [&ask](const Order& o) { return o.price == ask.price; });
            
            if (it != asks_.end()) {
                // Update quantity
                it->quantity = ask.quantity;
            } else {
                // Add new price level
                asks_.push_back(ask);
            }
        }
    }
    
    // Sort asks in ascending order by price
    std::sort(asks_.begin(), asks_.end(),
        [](const Order& a, const Order& b) { return a.price < b.price; });
}

double Orderbook::getBestBid() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bids_.empty() ? 0.0 : bids_[0].price;
}

double Orderbook::getBestAsk() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asks_.empty() ? 0.0 : asks_[0].price;
}

double Orderbook::getMidPrice() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return (bids_[0].price + asks_[0].price) / 2.0;
}

double Orderbook::getSpread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return asks_[0].price - bids_[0].price;
}

// OrderbookFetcher implementation
OrderbookFetcher::OrderbookFetcher() {
    // Pre-allocate memory for bid and ask updates
    bid_updates_.reserve(INITIAL_ORDERS_CAPACITY);
    ask_updates_.reserve(INITIAL_ORDERS_CAPACITY);
    
    // Initialize WebSocket client
    client_ = std::make_unique<websocket_client>();
    
    // Turn off logging
    client_->set_access_channels(websocketpp::log::alevel::none);
    client_->set_error_channels(websocketpp::log::elevel::fatal);
    
    // Set up TLS support
    client_->set_tls_init_handler([](websocketpp::connection_hdl) {
        return websocketpp::lib::make_shared<websocketpp::lib::asio::ssl::context>(
            websocketpp::lib::asio::ssl::context::tlsv12_client);
    });
    
    // Initialize ASIO
    client_->init_asio();
    
    // Set up message handlers
    client_->set_message_handler([this](websocketpp::connection_hdl hdl, websocket_client::message_ptr msg) {
        onMessage(client_.get(), hdl, msg);
    });
    
    client_->set_open_handler([this](websocketpp::connection_hdl hdl) {
        onOpen(client_.get(), hdl);
    });
    
    client_->set_close_handler([this](websocketpp::connection_hdl hdl) {
        onClose(client_.get(), hdl);
    });
    
    client_->set_fail_handler([this](websocketpp::connection_hdl hdl) {
        onFail(client_.get(), hdl);
    });
}

OrderbookFetcher::OrderbookFetcher(const std::string& symbol) : symbol_(symbol) {
    // Pre-allocate memory for bid and ask updates
    bid_updates_.reserve(INITIAL_ORDERS_CAPACITY);
    ask_updates_.reserve(INITIAL_ORDERS_CAPACITY);
    
    // Initialize WebSocket client
    client_ = std::make_unique<websocket_client>();
    
    // Turn off logging
    client_->set_access_channels(websocketpp::log::alevel::none);
    client_->set_error_channels(websocketpp::log::elevel::fatal);
    
    // Set up TLS support
    client_->set_tls_init_handler([](websocketpp::connection_hdl) {
        return websocketpp::lib::make_shared<websocketpp::lib::asio::ssl::context>(
            websocketpp::lib::asio::ssl::context::tlsv12_client);
    });
    
    // Initialize ASIO
    client_->init_asio();
    
    // Set up message handlers
    client_->set_message_handler([this](websocketpp::connection_hdl hdl, websocket_client::message_ptr msg) {
        onMessage(client_.get(), hdl, msg);
    });
    
    client_->set_open_handler([this](websocketpp::connection_hdl hdl) {
        onOpen(client_.get(), hdl);
    });
    
    client_->set_close_handler([this](websocketpp::connection_hdl hdl) {
        onClose(client_.get(), hdl);
    });
    
    client_->set_fail_handler([this](websocketpp::connection_hdl hdl) {
        onFail(client_.get(), hdl);
    });
}

OrderbookFetcher::~OrderbookFetcher() {
    disconnect();
}

bool OrderbookFetcher::connect(const std::string& exchange_url, const std::string& symbol) {
    if (running_) {
        std::cerr << "Already connected" << std::endl;
        return false;
    }
    
    symbol_ = symbol;
    
    try {
        // Start the client's perpetual service
        client_->start_perpetual();
        
        // Create connection
        websocketpp::lib::error_code ec;
        websocket_client::connection_ptr con = client_->get_connection(exchange_url, ec);
        
        if (ec) {
            std::cerr << "Could not create connection: " << ec.message() << std::endl;
            return false;
        }
        
        // Connect
        client_->connect(con);
        connection_ = con->get_handle();
        
        // Start websocket thread
        running_ = true;
        websocket_thread_ = std::thread(&OrderbookFetcher::run, this);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error connecting to exchange: " << e.what() << std::endl;
        return false;
    }
}

void OrderbookFetcher::disconnect() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    try {
        // Close connection
        if (client_ && connection_.lock()) {
            client_->close(connection_, websocketpp::close::status::normal, "Disconnecting");
        }
        
        // Stop client
        client_->stop_perpetual();
        
        // Notify thread to exit
        cv_.notify_all();
        
        // Join thread
        if (websocket_thread_.joinable()) {
            websocket_thread_.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error disconnecting from exchange: " << e.what() << std::endl;
    }
}

void OrderbookFetcher::registerCallback(OrderbookCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(callback);
}

bool OrderbookFetcher::subscribeToStream(const std::string& stream_name, StreamCallback callback) {
    if (!client_ || !connection_.lock()) {
        std::cerr << "Cannot subscribe to stream: not connected" << std::endl;
        return false;
    }
    
    try {
        // Store the callback
        {
            std::lock_guard<std::mutex> lock(stream_callbacks_mutex_);
            stream_callbacks_[stream_name] = callback;
        }
        
        // Create subscription message
        json subscription = {
            {"method", "SUBSCRIBE"},
            {"params", {stream_name}},
            {"id", 1}
        };
        
        // Get connection handle
        auto conn = connection_.lock();
        if (!conn) {
            std::cerr << "Cannot subscribe to stream: connection lost" << std::endl;
            return false;
        }
        
        // Send subscription message
        websocketpp::lib::error_code ec;
        client_->send(conn, subscription.dump(), websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Error sending subscription message: " << ec.message() << std::endl;
            return false;
        }
        
        std::cout << "Subscription message sent for stream: " << stream_name << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error subscribing to stream: " << e.what() << std::endl;
        return false;
    }
}

Orderbook OrderbookFetcher::getLatestOrderbook() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    return orderbook_;
}

void OrderbookFetcher::run() {
    try {
        // Run asio loop
        client_->run();
    } catch (const std::exception& e) {
        std::cerr << "Error in websocket thread: " << e.what() << std::endl;
    }
    
    // Wait for disconnect signal
    std::unique_lock<std::mutex> lock(callbacks_mutex_);
    cv_.wait(lock, [this] { return !running_; });
}

void OrderbookFetcher::onMessage(websocket_client* client, websocketpp::connection_hdl hdl, 
                                websocket_client::message_ptr msg) {
    try {
        // Parse the message
        std::string payload = msg->get_payload();
        json data = json::parse(payload);
        
        // Debug first few messages
        static int message_count = 0;
        if (message_count < 3) {
            std::cout << "Received message: " << payload.substr(0, 100) << "..." << std::endl;
            message_count++;
        }
        
        // Check if this is a stream message
        if (data.contains("stream") && data.contains("data")) {
            std::string stream = data["stream"];
            
            // Find the callback for this stream
            std::lock_guard<std::mutex> lock(stream_callbacks_mutex_);
            auto it = stream_callbacks_.find(stream);
            if (it != stream_callbacks_.end()) {
                // Call the callback with the data
                it->second(data["data"].dump());
                return;
            } else {
                std::cerr << "No callback registered for stream: " << stream << std::endl;
            }
        }
        // Check if this is a direct trade message (not wrapped in stream format)
        else if (data.contains("e") && (data["e"] == "aggTrade" || data["e"] == "trade")) {
            // Find the callback for the trade stream
            std::string stream_name = symbol_ + "@aggTrade";
            std::lock_guard<std::mutex> lock(stream_callbacks_mutex_);
            auto it = stream_callbacks_.find(stream_name);
            if (it != stream_callbacks_.end()) {
                // Call the callback with the data
                it->second(payload);
                return;
            }
        }
        // Check if this is a subscription response
        else if (data.contains("result") && data.contains("id")) {
            std::cout << "Subscription response: " << payload << std::endl;
            return;
        }
        // If not a stream message or subscription response, process as orderbook update
        else if (data.contains("lastUpdateId") || 
                (data.contains("e") && data["e"] == "depthUpdate")) {
            processOrderbookUpdate(data);
        }
        else {
            std::cerr << "Unknown message format: " << payload.substr(0, 100) << "..." << std::endl;
        }
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        std::cerr << "Raw message: " << msg->get_payload().substr(0, 100) << "..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
    }
}

void OrderbookFetcher::onOpen(websocket_client* client, websocketpp::connection_hdl hdl) {
    std::cout << "[" << std::chrono::system_clock::now() << "] [connect] Successful connection" << std::endl;
    
    try {
        // For Binance Futures, we need to subscribe to the depth stream for the symbol
        // The format is <symbol>@depth for the raw depth stream (same as spot but using the futures endpoint)
        // For perpetual contracts, the symbol should be lowercase, e.g., "btcusdt"
        std::string depth_stream = symbol_ + "@depth";
        std::string subscribe_msg = "{\"method\": \"SUBSCRIBE\", \"params\": [\"" + depth_stream + "\"], \"id\": 1}";
        
        std::cout << "Subscribing to " << depth_stream << " on Binance Futures" << std::endl;
        
        // Send subscription message with error checking
        websocketpp::lib::error_code ec;
        client->send(hdl, subscribe_msg, websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Error subscribing to orderbook: " << ec.message() << std::endl;
            return;
        }
        
        std::cout << "Connected to Binance Futures exchange and subscribed to " << depth_stream << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in onOpen handler: " << e.what() << std::endl;
    }
}

void OrderbookFetcher::onClose(websocket_client* client, websocketpp::connection_hdl hdl) {
    std::cout << "Disconnected from exchange" << std::endl;
}

void OrderbookFetcher::onFail(websocket_client* client, websocketpp::connection_hdl hdl) {
    std::cerr << "Connection failed" << std::endl;
}

void OrderbookFetcher::processOrderbookUpdate(const json& data) {
    try {
        // Clear pre-allocated vectors
        bid_updates_.clear();
        ask_updates_.clear();
        
        // Different exchanges have different message formats
        // This is for Binance
        if (data.contains("bids") && data.contains("asks")) {
            // Parse bids
            for (const auto& bid : data["bids"]) {
                if (bid.size() >= 2) {
                    double price = std::stod(bid[0].get<std::string>());
                    double quantity = std::stod(bid[1].get<std::string>());
                    bid_updates_.emplace_back(price, quantity);
                }
            }
            
            // Parse asks
            for (const auto& ask : data["asks"]) {
                if (ask.size() >= 2) {
                    double price = std::stod(ask[0].get<std::string>());
                    double quantity = std::stod(ask[1].get<std::string>());
                    ask_updates_.emplace_back(price, quantity);
                }
            }
            
            // Update orderbook
            {
                std::lock_guard<std::mutex> lock(orderbook_mutex_);
                orderbook_.updateBids(bid_updates_);
                orderbook_.updateAsks(ask_updates_);
            }
            
            // Notify callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : callbacks_) {
                    callback(orderbook_);
                }
            }
        }
        // For Binance depth update format
        else if (data.contains("e") && data["e"] == "depthUpdate") {
            // Parse bids
            if (data.contains("b")) {
                for (const auto& bid : data["b"]) {
                    if (bid.size() >= 2) {
                        double price = std::stod(bid[0].get<std::string>());
                        double quantity = std::stod(bid[1].get<std::string>());
                        bid_updates_.emplace_back(price, quantity);
                    }
                }
            }
            
            // Parse asks
            if (data.contains("a")) {
                for (const auto& ask : data["a"]) {
                    if (ask.size() >= 2) {
                        double price = std::stod(ask[0].get<std::string>());
                        double quantity = std::stod(ask[1].get<std::string>());
                        ask_updates_.emplace_back(price, quantity);
                    }
                }
            }
            
            // Update orderbook
            {
                std::lock_guard<std::mutex> lock(orderbook_mutex_);
                orderbook_.updateBids(bid_updates_);
                orderbook_.updateAsks(ask_updates_);
            }
            
            // Notify callbacks
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                for (const auto& callback : callbacks_) {
                    callback(orderbook_);
                }
            }
        }
        else {
            std::cerr << "Unknown orderbook update format" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing orderbook update: " << e.what() << std::endl;
    }
}

} // namespace trading 